import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
# from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
from utils.misc import get_rank

import models
from utils.ray_utils import get_rays
import systems
from typing import Dict, Any
from collections import OrderedDict
from systems.base import BaseSystem
from systems.criterions import PSNR
from systems import criterions
from models.utils import masked_mean

@systems.register('dnerf-system')
class DNeRFSystem(BaseSystem):
    def prepare(self):
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(**batch)
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        time_index = self.dataset.time_ids[index]
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            directions = self.dataset.directions[y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1)
            if self.dataset.all_depths is not None:
                batch['depth'] = self.dataset.all_depths[index, y, x].view(-1)
        else:
            c2w = self.dataset.all_c2w[index][0]
            directions = self.dataset.directions
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index].view(-1)
            if self.dataset.all_depths is not None:
                batch['depth'] = self.dataset.all_depths[index].view(-1)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background == 'black':
                self.model.background_color = torch.zeros((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])

        batch.update({
            'rays': rays,
            'time_index': time_index.squeeze().item(),
            'rgb': rgb,
            'mask': fg_mask,
            'near': self.dataset.near,
            'far': self.dataset.far,
        })

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor]):
        loss, stats = 0, OrderedDict()
        loss_weight = self.config.system.loss
        stats["loss_rgb"] = masked_mean((out["rgb"] - batch["rgb"]) ** 2, batch["mask"].reshape(-1, 1))
        loss += stats["loss_rgb"]
        stats["metric_psnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["mask"].reshape(-1, 1))

        if 'depth' in batch:
            stats["loss_depth"] = criterions.compute_depth_loss(batch["depth"], out["depth"])
            loss += loss_weight.depth*stats["loss_depth"]

        if self.training and loss_weight.dist > 0.0:
            pred_weights = out["weights"]  # nrays, n_samples, 1
            tvals = out["tvals"]  # nrays, n_samples, 1
            near, far = out['near'], out['far']
            pred_weights = pred_weights[:, :-1]
            svals = (tvals - near) / (far - near)
            stats["loss_dist"] = criterions.compute_dist_loss(pred_weights, svals)
            loss += loss_weight.dist*stats["loss_dist"]

        return loss, stats

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss, stats = self._level_fn(batch, out)
        for key, val in stats.items():
            self.log(f'train/{key}', val, prog_bar=True, rank_zero_only=True)

        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True)
        current_lr = self._update_learning_rate()
        self.log('train/current_lr', current_lr, prog_bar=True, rank_zero_only=True)

        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx, prefix='val'):
        out = self(batch)
        loss, stats = self._level_fn(batch, out)

        W, H = self.dataset.w, self.dataset.h
        img = self.save_image_grid(f"it{self.global_step}-{prefix}/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        stats['index'] = batch['index']
        stats['loss'] = loss
        if self.trainer.is_global_zero:
            self.logger.experiment.add_image(f'{prefix}/renderings', img/255., self.global_step, dataformats='HWC')
        return stats
          
    def validation_epoch_end(self, out, prefix='val'):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            metrics = [k for k in out[0].keys() if k.startswith('loss') or k.startswith('metric')]
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {k: step_out[k] for k in metrics}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {k: step_out[k][oi] for k in metrics}

            for key in metrics:
                m_val = torch.mean(torch.stack([o[key] for o in out_set.values()]))
                self.log(f'{prefix}/{key}', m_val, prog_bar=True, rank_zero_only=True)         
        return out

    def test_step(self, batch, batch_idx):  
        return self.validation_step(batch, batch_idx, prefix='test')
    
    def test_epoch_end(self, out):
        out = self.validation_epoch_end(out, prefix='test')
        if self.trainer.is_global_zero:
            can_mesh = self.model.isosurface(can_space=True)
            self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}_can.obj", can_mesh)
            
            mesh = self.model.isosurface(can_space=False)
            self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}.obj", mesh)

            idir = f"it{self.global_step}-test"
            self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=30)
