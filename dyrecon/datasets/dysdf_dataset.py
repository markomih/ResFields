import os

from glob import glob

import torch
import trimesh
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

# from torch.utils.data import Dataset, DataLoader, IterableDataset

import datasets
from utils.misc import get_rank
from utils.ray_utils import get_rays
# from . import utils
# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

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
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    return torch.stack(intrinsics_all), torch.stack(pose_all) # [n_images, 4, 4]

class DySDFDatasetBase():
    def setup(self, config, camera_list, split, load_time_steps=100000):
        # self.config = config
        print('Load data: Begin')
        self.split = split
        self.sampling = config.get('sampling', None)
        # data_ids = config.get(f'{split}_ids', -1)
        def _sample(data_list: list):
            ret = data_list #if data_ids == -1 else [data_list[i] for i in data_ids]
            return ret[:load_time_steps]
        self.camera_dict = {}
        _all_c2w, _all_images, _all_depths, _all_fg_masks, _frame_ids, _directions = [], [], [], [], [], []
        for cam_dir in camera_list:
            data_dir = os.path.join(config.data_root, cam_dir)
            if not os.path.exists(data_dir):
                raise FileNotFoundError(data_dir)
            print('Load data:', data_dir)
            _images_lis = sorted(glob(os.path.join(data_dir, 'rgb/*.png')))  
            _camera_dict = np.load(os.path.join(data_dir, 'cameras_sphere.npz'))
            world_mats_np = _sample([_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))]) # world_mat is a projection matrix from world to image
            scale_mats_np = _sample([_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(_images_lis))]) # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            intrinsics_all, pose_all = parse_cam(scale_mats_np, world_mats_np)

            images_lis = _sample(_images_lis)
            masks_lis = _sample(sorted(glob(os.path.join(data_dir, 'mask/*.png'))))

            images = torch.from_numpy(np.stack([cv.imread(im_name)[..., ::-1] for im_name in images_lis]) / 256.0)  # [n_images, H, W, 3]
            all_c2w = pose_all.float()[:, :3, :4]
            all_images = images.float()

            self.has_masks = len(masks_lis) > 0
            if self.has_masks:
                masks  = torch.from_numpy(np.stack([cv.imread(im_name) for im_name in masks_lis]) / 256.0)   # [n_images, H, W, 3]
                all_fg_masks = (masks > 0)[..., 0].float()
                all_images = all_images*all_fg_masks[..., None].float()
                _all_fg_masks.append(all_fg_masks)

            depth_lis = _sample(sorted(glob(os.path.join(data_dir, 'depth/*.png'))))
            self.has_depth = len(depth_lis) > 0
            if self.has_depth:
                depth_scale = config.get('depth_scale', 1000.)
                depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in depth_lis]) / depth_scale
                depths_np = depths_np*(1./scale_mats_np[0][0, 0])
                depths_np[depths_np == 0] = -1. # avoid nan values
                depths = torch.from_numpy(depths_np.astype(np.float32)).float()
                if self.has_masks:
                    depths[~(all_fg_masks > 0)] = -1

                _all_depths.append(torch.from_numpy(depths_np.astype(np.float32)).float())
                # depths_np[depths_np > 3.] = -1.
                # self.depths = torch.from_numpy(depths_np.astype(np.float32)).float().cuda()

            self.h, self.w = all_images.shape[1:-1]
            frame_ids = torch.tensor(list(range(all_images.shape[0]))).long()

            self.h, self.w = images.shape[1], images.shape[2]
            directions = get_ray_directions(self.h, self.w, intrinsics_all[0]) # (h, w, 3)
            directions = directions.unsqueeze(0).repeat(all_images.shape[0], 1, 1, 1) # [n_images, h, w, 3]
            self.intrinsics_inv = torch.inverse(intrinsics_all[0].float())
            
            _all_c2w.append(all_c2w)
            _all_images.append(all_images)
            _frame_ids.append(frame_ids)
            _directions.append(directions)
            self.camera_dict[cam_dir] = all_c2w[0]

        if self.split != 'train' and os.path.exists(os.path.join(config.data_root, 'cloud')):
            coud_dir = os.path.join(config.data_root, 'cloud')
            _mesh_lis = _sample(sorted(glob(os.path.join(coud_dir, '*.ply'))))
            self.clouds = [torch.from_numpy(trimesh.load(mesh_file).vertices).float() for mesh_file in _mesh_lis]
        else:
            self.clouds = None

        self.device = torch.device('cpu') #get_rank()
        self.all_c2w = torch.cat(_all_c2w, dim=0).to(self.device)
        self.all_images = torch.cat(_all_images, dim=0).to(self.device)
        self.frame_ids = torch.cat(_frame_ids, dim=0).to(self.device)
        self.directions = torch.cat(_directions, dim=0).to(self.device)
        self.image_pixels = self.h * self.w
        self.time_max = frame_ids.max() + 1
        # compute indices of foreground pixels for sampling
        if self.has_masks:
            self.all_fg_masks = torch.cat(_all_fg_masks, dim=0).to(self.device)
            self.fg_inds = torch.stack(torch.where((self.all_fg_masks > 0.0).bool()), -1)
            self.bg_inds = torch.stack(torch.where(~(self.all_fg_masks > 0.0).bool()), -1)
            yx_fg_mask = (self.all_fg_masks > 0.0).any(dim=0) # H, W
            self.yx_fg_inds = torch.stack(torch.where(yx_fg_mask), -1) # n_fg_pixels, 2
            self.yx_bg_inds = torch.stack(torch.where(~yx_fg_mask), -1) # n_bg_pixels, 2
        if self.has_depth:
            self.depths = torch.cat(_all_depths, dim=0).to(self.device)
        print('Load data: End', 'Shapes:', self.all_c2w.shape, self.all_images.shape, self.frame_ids.shape, self.directions.shape)

    def frame_id_to_time(self, frame_id):
        return (frame_id / self.time_max) * 2.0 - 1.0 # range of (-1, 1)

    def _sample_pixels(self):
        assert self.split == 'train'
        index, y, x = None, None, None
        train_num_rays = self.sampling.train_num_rays
        if self.sampling.strategy == 'balanced':
            fg_rays = train_num_rays//2
            bg_rays = int(train_num_rays - fg_rays)

            _fg_inds = self.fg_inds[torch.randint(0, self.fg_inds.shape[0], size=(fg_rays,), device=self.device)] # B,3
            _bg_inds = self.bg_inds[torch.randint(0, self.bg_inds.shape[0], size=(bg_rays,), device=self.device)] # B,3\
            _inds = torch.cat((_fg_inds, _bg_inds), 0)
            index, y, x = _inds[:, 0], _inds[:, 1], _inds[:, 2]
        elif self.sampling.strategy == 'fg_mask':
            _inds = self.fg_inds[torch.randint(0, self.fg_inds.shape[0], size=(train_num_rays,), device=self.device)] # B,3
            index, y, x = _inds[:, 0], _inds[:, 1], _inds[:, 2]
        elif self.sampling.strategy == 'time_balanced':
            n_cameras = self.all_images.shape[0]
            assert train_num_rays % n_cameras == 0, 'train_num_rays should be divisible by the number of cameras and frames'
            bg_rays = train_num_rays//4
            fg_rays = int(train_num_rays - bg_rays)

            _yx_fg_inds = self.yx_fg_inds[torch.randint(0, self.yx_fg_inds.shape[0], size=(fg_rays,), device=self.device)] # B,3
            _yx_bg_inds = self.yx_bg_inds[torch.randint(0, self.yx_bg_inds.shape[0], size=(bg_rays,), device=self.device)] # B,3\
            _yx_inds = torch.cat((_yx_fg_inds, _yx_bg_inds), 0)

            n_samples_per_time = train_num_rays//self.all_images.shape[0]
            index = torch.arange(0, n_cameras, device=self.device).view(-1, 1).expand(-1, n_samples_per_time).reshape(-1)
            y, x = _yx_inds[:, 0], _yx_inds[:, 1]
        else:
            index = torch.randint(0, len(self.all_images), size=(train_num_rays,), device=self.device)
            x = torch.randint(0, self.w, size=(train_num_rays,), device=self.device)
            y = torch.randint(0, self.h, size=(train_num_rays,), device=self.device)
        return index, y, x

    def sample_data(self, index=None):
        to_ret = {}
        if self.split == 'train':
            index, y, x = self._sample_pixels()
            directions = self.directions[index, y, x] # (B,3)
            c2w = self.all_c2w[index]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index, y, x].view(-1, self.all_images.shape[-1])
            if self.has_masks:
                to_ret['mask'] = self.all_fg_masks[index, y, x].view(-1) # n_rays
            if self.has_depth:
                to_ret['depth'] = self.depths[index, y, x].view(-1) # n_rays
        else:
            c2w = self.all_c2w[index].squeeze(0)
            directions = self.directions[index].squeeze(0) #(H,W,3)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index].view(-1, self.all_images.shape[-1])
            if self.has_masks:
                to_ret['mask'] = self.all_fg_masks[index].view(-1)
            if self.has_depth:
                to_ret['depth'] = self.depths[index].view(-1) # n_rays
        frame_id = self.frame_ids[index]
        rays_time = self.frame_id_to_time(frame_id).view(-1, 1)
        if rays_time.shape[0] != rays_o.shape[0]:
            rays_time = rays_time.expand(rays_o.shape[0], rays_time.shape[1])
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), rays_time], dim=-1)

        to_ret.update({
            'rays': rays, # n_rays, 7
            'frame_id': frame_id.squeeze(), # n_rays
            'rgb': rgb, # n_rays, 3
        })
        return to_ret

class DySDFDataset(torch.utils.data.Dataset, DySDFDatasetBase):
    def __init__(self, config, camera_list, split, load_time_steps=100000):
        self.setup(config, camera_list, split, load_time_steps)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        batch = self.sample_data(index)
        batch.update(dict(index=index))
        if self.clouds is not None:
            batch['cloud'] = self.clouds[index]
        return batch

class DySDFPredictDataset(torch.utils.data.Dataset, DySDFDatasetBase):
    def __init__(self, config, camera_list, split, load_time_steps=100000):
        self.setup(config, camera_list, split, load_time_steps)
        cam_1 = config.predict.cam_1
        cam_2 = config.predict.cam_2
        n_imgs = config.predict.n_imgs
        self.rays_list, self._frame_list = self.interpolate_cameras(cam_1, cam_2, n_imgs)

    def interpolate_cameras(self, cam_1, cam_2, n_imgs):
        pose_1 = self.camera_dict[cam_1]
        pose_2 = self.camera_dict[cam_2]
        frame_list = torch.cat((self.frame_ids, torch.flip(self.frame_ids, [0])[1:])).view(1, -1).repeat(100, 1).view(-1)[:n_imgs]
        rays_list = []
        for i in range(n_imgs):
            ratio = np.sin(((i / n_imgs) - 0.5) * np.pi) * 0.5 + 0.5
            rays_time = self.frame_id_to_time(torch.tensor(frame_list[i]))
            rays_list.append(self.gen_rays_between(pose_1, pose_2, ratio, time_step=rays_time))
        return rays_list, frame_list

    def gen_rays_between(self, pose_0: np.ndarray, pose_1: np.ndarray, ratio:float, time_step=0):
        _pose_0, _pose_1 = np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([1.0, 1.0, 1.0, 1.0])
        _pose_0[:3], _pose_1[:3] = pose_0[:3], pose_1[:3]

        RT0, RT1 = np.linalg.inv(_pose_0), np.linalg.inv(_pose_1)
        T = RT0[:3, 3]*(1.0 - ratio) + RT1[:3, 3]*ratio

        rots = Rot.from_matrix(np.stack([RT0[:3,:3], RT1[:3,:3]]))
        slerp = Slerp([0, 1], rots)
        rot = slerp(ratio)

        RT = np.diag([1.0, 1.0, 1.0, 1.0])
        RT[:3, :3] = rot.as_matrix()
        RT[:3, 3] = T
        c2w = np.linalg.inv(RT)
        c2w = torch.from_numpy(c2w).float()[:3, :4]

        directions = self.directions[0].view(-1, 3)
        rays_o, rays_d = get_rays(directions, c2w[None].expand(directions.shape[0], -1, -1)[:,:3,:4])
        rays_time = torch.full_like(rays_o[..., :1], fill_value=time_step)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1), rays_time], dim=-1)
        return rays

    def __len__(self):
        return len(self.rays_list)
    
    def __getitem__(self, index):
        batch = {
            'rays': self.rays_list[index].view(-1, self.rays_list[index].shape[-1]),
            'index': index,
            'frame_id': self._frame_list[index],
        }
        return batch

class DySDFIterableDataset(torch.utils.data.IterableDataset, DySDFDatasetBase):
    def __init__(self, config, camera_list, split, load_time_steps=100000):
        self.setup(config, camera_list, split, load_time_steps)

    def __iter__(self):
        while True:
            batch = self.sample_data(None)
            yield batch

@datasets.register('dysdf_dataset')
class DySDFDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        load_time_steps =self.config.get('load_time_steps', 100000)
        if stage in [None, 'fit']:
            self.train_dataset = DySDFIterableDataset(self.config, self.config.train_split, 'train', load_time_steps)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DySDFDataset(self.config, self.config.val_split, 'test', self.config.get('val_load_time_steps', load_time_steps))
        if stage in [None, 'test']:
            self.test_dataset = DySDFDataset(self.config, self.config.test_split, 'test', self.config.get('test_load_time_steps', load_time_steps))
        if stage in [None, 'predict']:
            self.predict_dataset = DySDFPredictDataset(self.config, self.config.train_split, 'train', load_time_steps)

    @staticmethod
    def get_metadata(config):
        aabb = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        n_imgs = glob(os.path.join(config.data_root, config.train_split[0], 'rgb', '*.png'))
        n_frames = min(config.load_time_steps, len(n_imgs))
        if os.path.exists(os.path.join(config.data_root, 'aabb.txt')):
            with open(os.path.join(config.data_root, 'aabb.txt'), 'r') as f:
                aabb = f.read().strip()
            aabb = eval(aabb)
        return {
            'scene_aabb': aabb,
            'n_frames': n_frames,
        }

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return torch.utils.data.DataLoader(
            dataset, 
            num_workers=6,#os.cpu_count(), 
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
