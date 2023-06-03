import os

from glob import glob

import torch
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, IterableDataset

import datasets
from utils.misc import get_rank

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

class DysurfDatasetBase():

    def setup(self, config, split):
        self.config = config
        self.rank = get_rank()

        print('Load data: Begin')
        self.conf = config
        load_time_steps = self.config.get('load_time_steps', 100000)
        data_ids = self.config.get(f'{split}_ids', -1)
        def _sample(data_list: list):
            ret = data_list if data_ids == -1 else [data_list[i] for i in data_ids]
            return ret[:load_time_steps]
        self.data_dir = self.config.root_dir

        _images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))  
        _camera_dict = np.load(os.path.join(self.data_dir, 'cameras_sphere.npz'))
        # world_mat is a projection matrix from world to image
        world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))])
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))])
        self.intrinsics_all, self.pose_all = parse_cam(scale_mats_np, world_mats_np)

        self.images_lis = _sample(_images_lis)
        self.masks_lis = _sample(sorted(glob(os.path.join(self.data_dir, 'mask/*.png'))))
        self.depth_lis = _sample(sorted(glob(os.path.join(self.data_dir, 'depth/*.png'))))
        if len(glob(os.path.join(self.data_dir, 'normal/*.png'))) > 0:
            self.normal_lis = _sample(sorted(glob(os.path.join(self.data_dir, 'normal/*.png'))))
            self.normals = self._load_normal()
        if len(glob(os.path.join(self.data_dir, 'optflow/*.png'))) > 0:
            self.optflow_lis = _sample(sorted(glob(os.path.join(self.data_dir, 'optflow/*.npy'))))
        self.time_index = torch.tensor(list(map(lambda x: int(os.path.splitext(os.path.basename(x))[0]), self.images_lis))).long()

        self.images = torch.from_numpy(np.stack([cv.imread(im_name)[..., ::-1] for im_name in self.images_lis]) / 256.0)  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0)   # [n_images, H, W, 3]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]

        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.depths = self._load_depth(scale=1./scale_mats_np[0][0, 0])

        print('Load data: End')
        self.all_c2w = self.pose_all.float().to(self.rank)[:, :3, :4]
        self.all_images = self.images.float().to(self.rank)
        self.all_fg_masks = (self.masks > 0).to(self.rank)[..., 0].float()
        self.all_images = self.all_images*self.all_fg_masks[..., None].float()
        self.all_depths = self.depths.to(self.rank) if self.has_depth() else None

        self.h, self.w = self.all_images.shape[1:-1]
        self.time_ids = torch.tensor(list(range(self.all_images.shape[0]))).to(self.rank)
        self.time_ids_torch = self.time_ids.long()
        self.dynamic_pixel_masks = None
        self.covisible_masks = None
        self.near = 0.5
        self.far = 5.0

        self.directions = get_ray_directions(self.h, self.w, self.intrinsics_all[0]).to(self.rank) # (h, w, 3)
        print('SHAPES:', self.all_c2w.shape, self.all_images.shape, self.all_fg_masks.shape, self.directions.shape)

    def has_depth(self):
        return len(self.depth_lis) > 0

    def has_normal(self):
        return len(self.normal_lis) > 0

    def has_optflow(self):
        return len(self.optflow_lis) > 0

    def _load_depth(self, scale):
        if self.has_depth():
            depth_scale = 1000.
            depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.depth_lis]) / depth_scale
            depths_np = depths_np*scale
            depths_np[depths_np == 0] = -1. # avoid nan values
            depths = torch.from_numpy(depths_np.astype(np.float32)).float().cpu()
            return depths

    def _load_normal(self):
        if self.has_normal():
            normals_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.normal_lis])/255.0
            normals_np = 2.0*(normals_np-0.5) # [0,1.]-> [-1,1]
            normals = torch.from_numpy(normals_np.astype(np.float32)).float().cpu()
            return normals


class DysurfDataset(Dataset, DysurfDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class DysurfIterableDataset(IterableDataset, DysurfDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}

class DysurfPredictDataset(Dataset, DysurfDatasetBase):
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

@datasets.register('dysurf_dataset')
class DysurfDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DysurfIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DysurfDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = DysurfDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = DysurfPredictDataset(self.config, self.config.train_split)

    @staticmethod
    def get_metadata(config):
        return {
            'near': 0.1, 'far': 5.0,
            'time_max': 100,
            'train_img_hw': (512, 512),  # TODO: the resolution is temporary hardcoded
            # 'bkgd_points': bkgd_points,
            'scene_aabb': [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        }

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
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
