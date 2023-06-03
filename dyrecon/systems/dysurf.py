import torch
import trimesh
import torch.nn.functional as F
import numpy as np
from utils.ray_utils import get_rays

import systems
import models
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM


@systems.register('dysurf-system')
class DysurfSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
        }
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
        # if stage == "test":
        #     import pdb; pdb.set_trace()
        time_index = self.dataset.time_ids[index]
        has_depth = self.dataset.all_depths is not None
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
            if has_depth:
                batch['depth'] = self.dataset.all_depths[index, y, x].view(-1)
        else:
            c2w = self.dataset.all_c2w[index][0]
            directions = self.dataset.directions
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index].view(-1)
            if has_depth:
                batch['depth'] = self.dataset.all_depths[index].view(-1) if self.dataset.all_depths is not None else None

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'time_index': time_index,
            'index': index,
            'near': getattr(self.dataset, "near", None),
            'far': getattr(self.dataset, "far", None),
        })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        true_rgb = batch['rgb']
        time_index = batch['time_index'].squeeze()

        color_fine = out['comp_rgb']
        s_val = out['s_val']
        eikonal_loss = out['gradient_o_error']
        weight_sum = out['opacity']
        depth_map = out['depth'].reshape(-1)
        # normal_map = out['normal']

        near, far = out['near'].squeeze(), out['far'].squeeze()
        rays_o, rays_d = batch['rays'][:, :3], batch['rays'][:, 3:6]

        # mask loss
        mask = (batch['fg_mask'][:, None] > 0.5).float()
        mask_sum = mask.sum() + 1e-5

        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        lw = self.config.system.loss
        loss_dict = {
            'color_loss': color_fine_loss*lw.color_weight, 
            'mask_loss': mask_loss*lw.mask_weight, 
            'eikonal_loss': eikonal_loss*lw.igr_weight,
        }
        depth_loss, angle_loss, depth_sdf_loss, depth_eik_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0.), torch.tensor(0.)
        if mask_sum > 1e-5 and 'depth' in batch:
            rays_depth = batch['depth']
            valid_depth_region = (rays_depth > 0.) & (rays_depth > near) & (mask.squeeze() > 0.) & (rays_depth < far)
            if valid_depth_region.any():
                depth_loss = F.l1_loss(depth_map[valid_depth_region], rays_depth[valid_depth_region], reduction='mean')

                depth_pts = (rays_o+rays_depth[:, None]*rays_d)[valid_depth_region]
                angle_loss, depth_sdf_loss, depth_eik_loss = self.model.get_angle_error(rays_d[valid_depth_region], depth_pts, rays_time=time_index)

                loss_dict.update({
                    'depth_loss': depth_loss*lw.geo_weight,
                    'angle_loss': angle_loss*lw.angle_weight,
                    'depth_sdf_loss': depth_sdf_loss*lw.depth_sdf_weight,
                    'depth_eik_loss': depth_eik_loss*lw.depth_eik_weight,
                })

        for key, val in loss_dict.items():
            self.log(f'train/{key}', val, rank_zero_only=True)

        loss = sum(loss_dict.values())
        current_lr = self.update_learning_rate()

        # log other statistics
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        self.log('train/lr', current_lr, prog_bar=True, rank_zero_only=True)
        self.log('train/psnr', psnr, prog_bar=True, rank_zero_only=True)
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True)
        self.log('train/s_val', s_val, rank_zero_only=True)

        if (self.config.system.vis_mesh_every > 0) and (self.global_step % self.config.system.vis_mesh_every == 0):
            if self.trainer.is_global_zero:
                can_mesh = self.model.isosurface(time_index=0, can_space=True, resolution=128)
                self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}_can.obj", can_mesh)
                
                mesh = self.model.isosurface(time_index=0, can_space=False, resolution=128)
                self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}.obj", mesh)

        return {
            'loss': loss
        }
    
    def update_learning_rate(self):
        iter_step = self.global_step
        warm_up_end = self.config.system.scheduler.warm_up_end
        max_steps = self.config.system.scheduler.max_steps
        learning_rate = self.config.system.scheduler.learning_rate
        learning_rate_alpha = self.config.system.scheduler.learning_rate_alpha
        if iter_step < warm_up_end:
            learning_factor = 1.0
        else:
            alpha = learning_rate_alpha
            progress = (iter_step - warm_up_end) / (max_steps - warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        optimizer = self.optimizers()
        for g in optimizer.param_groups:
            g['lr'] = learning_rate * learning_factor
        return learning_rate*learning_factor
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        # ssim = self.criterions['ssim'](out['comp_rgb'], batch['rgb'])
        depth_l1_loss =F.l1_loss(out['depth'], batch.get('depth', out['depth']))
        W, H = self.dataset.w, self.dataset.h
        img = out['comp_rgb'].view(H, W, 3)
        opacity = torch.clamp(out['opacity'].view(H, W), 0., 1.)
        depth = out['depth'].view(H, W)
        normal = 0.5 + 0.5*out['normal'].view(H, W, 3)
        img_path = f"it{self.global_step}-{batch['time_index'][0].item()}.png"
        img = self.save_image_grid(img_path, [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': normal*opacity[..., None], 'kwargs': {'data_format': 'HWC'}},
            {'type': 'depth', 'img': depth*opacity, 'kwargs': {'depth_max': 3.0}},
            {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        if self.trainer.is_global_zero:
            self.logger.experiment.add_image(f'val/renderings', img/255., self.global_step, dataformats='HWC')
        return {
            'psnr': psnr,
            # 'ssim': ssim,
            'depth_l1_loss': depth_l1_loss,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def _log_eval(self, out, prefix='val'):
        if self.trainer.is_global_zero:
            metric_keys = ['psnr', 'depth_l1_loss'] # 'ssim',
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    # out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    out_set[step_out['index'].item()] = {key: step_out[key] for key in metric_keys}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        # out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        out_set[index[0].item()] = {key: step_out[key][oi] for key in metric_keys}

            for metric_key in metric_keys:
                val = torch.mean(torch.stack([o[metric_key] for o in out_set.values()]))
                self.log(f'{prefix}/{metric_key}', val, prog_bar=True, rank_zero_only=True)


    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        self._log_eval(out, prefix='val')

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        # ssim = self.criterions['ssim'](out['comp_rgb'], batch['rgb'])
        depth_l1_loss =F.l1_loss(out['depth'], batch.get('depth', out['depth']))
        W, H = self.config.dataset.img_wh
        img = out['comp_rgb'].view(H, W, 3)
        depth = out['depth'].view(H, W)
        normal = 0.5 + 0.5*out['normal'].view(H, W, 3)
        opacity = torch.clamp(out['opacity'].view(H, W), 0., 1.)
        img_path = f"it{self.global_step}-test/{batch['time_index'][0].item()}.png"
        img = self.save_image_grid(img_path, [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': normal*opacity[..., None], 'kwargs': {'data_format': 'HWC'}},
            {'type': 'depth', 'img': depth*opacity, 'kwargs': {'depth_max': 3.0}},
            {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        if self.trainer.is_global_zero:
            self.logger.experiment.add_image(f'test/renderings', img/255., self.global_step, dataformats='HWC')
        return {
            'psnr': psnr,
            # 'ssim': ssim,
            'depth_l1_loss': depth_l1_loss,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            self._log_eval(out, prefix='test')

            idir = f"it{self.global_step}-test"
            self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=30)
            
            can_mesh = self.model.isosurface(time_index=0, can_space=True)
            self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}_can.obj", can_mesh)
            
            mesh = self.model.isosurface(time_index=0, can_space=False)
            self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}.obj", mesh)
