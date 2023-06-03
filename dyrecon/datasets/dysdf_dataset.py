import os

from glob import glob

import torch
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

# from torch.utils.data import Dataset, DataLoader, IterableDataset

import datasets
from utils.misc import get_rank, vector_cat

from . import utils

def get_ray_directions(H, W, K, OPENGL_CAMERA=False):
    x, y = torch.meshgrid(
        torch.arange(W, device=K.device),
        torch.arange(H, device=K.device),
        indexing="xy",
    )
    camera_dirs = F.pad(torch.stack([
            (x - K[0, 2] + 0.5) / K[0, 0],
            (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
        ], dim=-1,), (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]

    return camera_dirs

def parse_cam(scale_mats_np, world_mats_np):
    intrinsics_all, pose_all = [], []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = utils.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    return torch.stack(intrinsics_all), torch.stack(pose_all) # [n_images, 4, 4]

class DySDFDatasetBase():
    def __init__(self, config):
        # self.config = config
        self.rank = get_rank()

        print('Load data: Begin')
        load_time_steps = config.get('load_time_steps', 100000)
        # data_ids = config.get(f'{split}_ids', -1)
        def _sample(data_list: list):
            ret = data_list #if data_ids == -1 else [data_list[i] for i in data_ids]
            return ret[:load_time_steps]
        data_dir = config.root_dir

        _images_lis = sorted(glob(os.path.join(data_dir, 'rgb/*.png')))  
        _camera_dict = np.load(os.path.join(data_dir, 'cameras_sphere.npz'))
        # world_mat is a projection matrix from world to image
        world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))])
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))])
        intrinsics_all, pose_all = parse_cam(scale_mats_np, world_mats_np)

        images_lis = _sample(_images_lis)
        masks_lis = _sample(sorted(glob(os.path.join(data_dir, 'mask/*.png'))))

        images = torch.from_numpy(np.stack([cv.imread(im_name)[..., ::-1] for im_name in images_lis]) / 256.0)  # [n_images, H, W, 3]
        masks  = torch.from_numpy(np.stack([cv.imread(im_name) for im_name in masks_lis]) / 256.0)   # [n_images, H, W, 3]
        # intrinsics_all_inv = torch.inverse(intrinsics_all)  # [n_images, 4, 4]

        self.h, self.w = images.shape[1], images.shape[2]
        self.image_pixels = self.H * self.W

        print('Load data: End')
        self.all_c2w = pose_all.float().to(self.rank)[:, :3, :4]
        self.all_images = images.float().to(self.rank)
        self.all_fg_masks = (masks > 0).to(self.rank)[..., 0].float()
        self.all_images = self.all_images*self.all_fg_masks[..., None].float()

        self.h, self.w = self.all_images.shape[1:-1]
        self.time_ids = torch.tensor(list(range(self.all_images.shape[0]))).to(self.rank).long()
        self.near = None
        self.far = None

        self.directions = get_ray_directions(self.h, self.w, intrinsics_all[0]).to(self.rank) # (h, w, 3)

        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]).to(self.rank)
        if os.path.exists(os.path.join(data_dir, 'aabb.npy')):
            self.aabb = torch.from_numpy(np.load(os.path.join(data_dir, 'aabb.npy'))).to(self.rank)

        print('SHAPES:', self.all_c2w.shape, self.all_images.shape, self.all_fg_masks.shape, self.directions.shape)

    @staticmethod
    def merge_datasets(ds_list: list):
        if len(ds_list) == 1:
            return ds_list[0]

        attr_list = [
            'all_c2w', 'all_images', 'all_fg_masks', 'time_ids', 'directions', 
        ]

        ds = ds_list[0]
        for attr in attr_list:
            val_list = [getattr(_d, attr) for _d in ds_list]
            if isinstance(val_list[0], list) or isinstance(val_list[0], np.ndarray) or torch.is_tensor(val_list[0]):
                val = vector_cat(val_list, dim=0)
            elif isinstance(val_list[0], dict):
                val = {k: vector_cat([v[k] for v in val_list]) for k in val_list[0].keys()}
            elif val_list[0] is None:
                val = None
            else:
                raise NotImplementedError
            setattr(ds, attr, val)
        ds.n_views = len(ds_list)
        ds.n_images = ds.images_np.shape[0]
        return ds

    @classmethod
    def from_conf(cls, conf):
        train_cam_list = conf.train_cam_list
        val_cam = conf.val_cam

        # create a list of datasets
        train_ds_list = [DySDFDatasetBase(conf, os.path.join(conf.data_dir, train_cam)) for train_cam in train_cam_list]
        if isinstance(val_cam, list):
            val_ds_list = [DySDFDatasetBase(conf, os.path.join(conf.data_dir, _vc)) for _vc in val_cam]
            val_ds = DySDFDatasetBase.merge_datasets(val_ds_list)
        else:
            val_ds = DySDFDatasetBase(conf, os.path.join(conf.data_dir, val_cam))

        # merge datasets
        train_ds = DySDFDatasetBase.merge_datasets(train_ds_list)

        return train_ds, val_ds


class DySDFDataset(torch.utils.data.Dataset, DySDFDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class DySDFIterableDataset(torch.utils.data.IterableDataset, DySDFDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}

class DySDFPredictDataset(torch.utils.data.Dataset, DySDFDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def setup(self, config, split):
        super().setup(config, split)
        # create new cameras
        cams = self.get_360cams(self.w, self.h)

        # self.rank = _get_rank()
        self.all_c2w = torch.stack([c['c2ws'] for c in cams]).float().to(self.rank)[:, :3, :4]

    def __len__(self):
        return self.all_c2w.shape[0]
    
    def __getitem__(self, index):
        return {
            'index': index
        }

@datasets.register('dysdf_dataset')
class DySDFDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DySDFIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DySDFDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = DySDFDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = DySDFPredictDataset(self.config, self.config.train_split)

    @staticmethod
    def get_metadata(config):
        return dict()
        # return {
        #     'near': 0.1, 'far': 5.0,
        #     'time_max': 100,
        #     'train_img_hw': (512, 512),  # TODO: the resolution is temporary hardcoded
        #     # 'bkgd_points': bkgd_points,
        #     'scene_aabb': [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        # }

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return torch.utils.data.DataLoader(
            dataset, 
            num_workers=0,#os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
