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
    def setup(self, config, camera_list):
        # self.config = config
        print('Load data: Begin')
        load_time_steps = config.get('load_time_steps', 100000)
        # data_ids = config.get(f'{split}_ids', -1)
        def _sample(data_list: list):
            ret = data_list #if data_ids == -1 else [data_list[i] for i in data_ids]
            return ret[:load_time_steps]
        
        _all_c2w, _all_images, _all_fg_masks, _frame_ids, _directions = [], [], [], [], []
        for cam_dir in camera_list:
            data_dir = os.path.join(config.data_root, cam_dir)
            if not os.path.exists(data_dir):
                raise FileNotFoundError(data_dir)

            _images_lis = sorted(glob(os.path.join(data_dir, 'rgb/*.png')))  
            _camera_dict = np.load(os.path.join(data_dir, 'cameras_sphere.npz'))
            world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))]) # world_mat is a projection matrix from world to image
            scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))]) # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            intrinsics_all, pose_all = parse_cam(scale_mats_np, world_mats_np)

            images_lis = _sample(_images_lis)
            masks_lis = _sample(sorted(glob(os.path.join(data_dir, 'mask/*.png'))))

            images = torch.from_numpy(np.stack([cv.imread(im_name)[..., ::-1] for im_name in images_lis]) / 256.0)  # [n_images, H, W, 3]
            masks  = torch.from_numpy(np.stack([cv.imread(im_name) for im_name in masks_lis]) / 256.0)   # [n_images, H, W, 3]
            # intrinsics_all_inv = torch.inverse(intrinsics_all)  # [n_images, 4, 4]

            all_c2w = pose_all.float()[:, :3, :4]
            all_images = images.float()
            all_fg_masks = (masks > 0)[..., 0].float()
            all_images = all_images*all_fg_masks[..., None].float()

            self.h, self.w = all_images.shape[1:-1]
            frame_ids = torch.tensor(list(range(all_images.shape[0]))).long()

            self.h, self.w = images.shape[1], images.shape[2]
            directions = get_ray_directions(self.h, self.w, intrinsics_all[0]) # (h, w, 3)
            directions = directions.unsqueeze(0).repeat(all_images.shape[0], 1, 1, 1) # [n_images, h, w, 3]
            
            _all_c2w.append(all_c2w)
            _all_images.append(all_images)
            _all_fg_masks.append(all_fg_masks)
            _frame_ids.append(frame_ids)
            _directions.append(directions)

        rank = get_rank()
        self.all_c2w = torch.cat(_all_c2w, dim=0).to(rank)
        self.all_images = torch.cat(_all_images, dim=0).to(rank)
        self.all_fg_masks = torch.cat(_all_fg_masks, dim=0).to(rank)
        self.frame_ids = torch.cat(_frame_ids, dim=0).to(rank)
        self.directions = torch.cat(_directions, dim=0).to(rank)
        self.image_pixels = self.h * self.w
        self.time_max = frame_ids.max() + 1
        print('Load data: End', 'Shapes:', self.all_c2w.shape, self.all_images.shape, self.all_fg_masks.shape, self.frame_ids.shape, self.directions.shape)

    def frame_id_to_time(self, frame_id):
        return (frame_id / self.time_max) * 2.0 - 1.0 # range of (-1, 1)

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
        aabb = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        if os.path.exists(os.path.join(config.data_root, 'aabb.npy')):
            aabb = np.loadtxt(os.path.join(config.data_root, 'aabb.npy')).tolist()
        return {
            'scene_aabb': aabb,
        }
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
