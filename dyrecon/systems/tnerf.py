import torch
import torch.nn.functional as F
import numpy as np
import yaml

from utils.ray_utils import get_rays
import systems
from typing import Dict, Any
from collections import OrderedDict
from systems.base import BaseSystem
from systems import criterions
from models.utils import masked_mean
from utils.misc import plot_heatmap

@systems.register('tnerf-system')
class TNeRFSystem(BaseSystem):
    def prepare(self):
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        # TODO here use model.render
        return self.model(**batch) 

    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        time_index = self.dataset.time_ids_torch[index]

        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            directions = self.dataset.directions[y, x] # (B,3)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1)
            if self.dataset.all_depths is not None:
                batch['depth'] = self.dataset.all_depths[index, y, x].view(-1)
            batch['index'] = index
            batch['y'] = y
            batch['x'] = x
            if self.dataset.dynamic_pixel_masks is not None:
                batch['dynamic_pixel_masks'] = self.dataset.dynamic_pixel_masks[index, y, x].view(-1)
        else:
            c2w = self.dataset.all_c2w[index][0]
            directions = self.dataset.directions #(H,W,3)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index].view(-1)
            if self.dataset.all_depths is not None:
                batch['depth'] = self.dataset.all_depths[index].view(-1)

            if self.dataset.covisible_masks is not None:
                batch['covisible_masks'] = self.dataset.covisible_masks[index].view(-1)

            if self.dataset.dynamic_pixel_masks is not None:
                batch['dynamic_pixel_masks'] = self.dataset.dynamic_pixel_masks[index].view(-1)
        
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
        setattr(self.model, 'global_step', self.global_step)
        batch.update({
            'rays': rays,
            'time_index': time_index.squeeze(),
            'rgb': rgb,
            'mask': fg_mask,
            'near': self.dataset.near,
            'far': self.dataset.far,
        })

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor], level_name: str):
        loss, stats = 0, OrderedDict()
        loss_weight = self.config.system.loss
        stats["loss_rgb"] = masked_mean((out["rgb"] - batch["rgb"]) ** 2, batch["mask"].reshape(-1, 1))
        loss += stats["loss_rgb"]
        stats["metric_psnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["mask"].reshape(-1, 1))
        if 'covisible_masks' in batch:
            stats["metric_mpsnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["covisible_masks"].reshape(-1, 1))
        if not self.training and 'dynamic_pixel_masks' in batch:
            stats["metric_dynamic_mpsnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["dynamic_pixel_masks"].reshape(-1, 1))

        # # TODO! consider adding back the mask loss
        # if 'mask' in batch:
        #     mask_loss = F.binary_cross_entropy(out['opacity'].clip(1e-3, 1.0 - 1e-3), (batch['mask']> 0.5).float())
        # TODO: add eikonal loss
        if 'sdf_grad_samples' in out:
            stats["loss_eikonal"] = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
            loss += loss_weight.eikonal * stats["loss_eikonal"]


        if 'depth' in batch:
            stats["loss_depth"] = criterions.compute_depth_loss(batch["depth"], out["depth"])
            loss += loss_weight.depth*stats["loss_depth"]

        if self.training and loss_weight.get('dist', 0.0) > 0.0:
            pred_weights = out["weights"].squeeze(-1)  # nrays, n_samples
            tvals = out["tvals"].squeeze(-1)  # nrays, n_samples
            near, far = batch['near'], batch['far']
            pred_weights = pred_weights[:, :-1]
            svals = (tvals - near) / (far - near)
            stats["loss_dist"] = criterions.compute_dist_loss(pred_weights.unsqueeze(-1), svals.unsqueeze(-1))
            loss += loss_weight.dist*stats["loss_dist"]

        return loss, stats

    def training_step(self, batch, batch_idx):
        out = self(batch)
        # regularizations = self.model.regularization_loss(batch)

        losses = dict()
        self.levels = out.keys()
        for _level in self.levels:
            losses[_level], stats = self._level_fn(batch, out[_level], _level)
            for key, val in stats.items():
                self.log(f'train/{_level}_{key}', val, prog_bar=True, rank_zero_only=True)

        loss = sum(losses.values())
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True)
        current_lr = self._update_learning_rate()
        # self.log('train/current_lr', current_lr, prog_bar=True, rank_zero_only=True)

        return {
            'loss': loss
        }

    def vis_extra_images(self, batch, out):
        return []

    def validation_step(self, batch, batch_idx, prefix='val'):
        out = self(batch) #rays (N, 6), rgb (N,3), depth (N,1)
        W, H = self.dataset.w, self.dataset.h

        stats_dict = dict()
        self.levels = out.keys()
        for _level in self.levels:
            loss, stats = self._level_fn(batch, out[_level], _level)
            if 'covisible_masks' in batch:
                _out = out[_level]
                stats["metric_mssim"] = criterions.compute_ssim(_out["rgb"].reshape((H,W,3)).cpu(), batch["rgb"].reshape((H,W,3)), batch["covisible_masks"].reshape(H, W, 1).cpu())

            stats_dict.update({f"{_level}_{key}": val for key, val in stats.items()})
            stats_dict[f'{_level}_loss'] = loss
            
            _log_imgs = [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out[_level]['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ]
            _log_imgs += self.vis_extra_images(batch, out[_level])
            if 'normal' in out[_level]:
                normal = 0.5 + 0.5*out[_level]['normal'].view(H, W, 3)*out[_level]['opacity'].view(H, W, 1)
                _log_imgs.append(
                    {'type': 'rgb', 'img': normal, 'kwargs': {'data_format': 'HWC'}},
                )

            _log_imgs += [
                {'type': 'grayscale', 'img': out[_level]['depth'].view(H, W), 'kwargs': {}},
                {'type': 'grayscale', 'img': out[_level]['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            ]

            img = self.save_image_grid(f"it{self.global_step}-{prefix}_{_level}/{batch['index'][0].item()}.png", _log_imgs)
            if self.trainer.is_global_zero:
                if self.logger is not None:
                    self.logger.experiment.add_image(f'{prefix}/{_level}_renderings', img/255., self.global_step, dataformats='HWC')

        if self.trainer.is_global_zero:  # VISUALIZE
            ambient_codes = getattr(self.model, 'ambient_codes', None)
            if ambient_codes is not None:
                img_ambient_codes = plot_heatmap(ambient_codes.clone().detach().cpu().numpy())
                if self.logger is not None:
                    self.logger.experiment.add_image(f'{prefix}/img_ambient_codes', img_ambient_codes/255., self.global_step, dataformats='HWC')

        stats_dict['index'] = batch['index']
        return stats_dict

    def validation_epoch_end(self, out, prefix='val'):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            metrics_dict = {}
            out_set = {}
            metrics = [k for k in out[0].keys() if 'loss' in k or 'metric' in k]
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
                metrics_dict[key] = float(m_val.detach().cpu())
                self.log(f'{prefix}/{key}', m_val, prog_bar=True, rank_zero_only=True)         
            return metrics_dict
        return out

    def test_step(self, batch, batch_idx):  
        return self.validation_step(batch, batch_idx, prefix='test')

    def test_epoch_end(self, out):
        prefix = 'test'
        metrics_dict = self.validation_epoch_end(out, prefix=prefix)
        if self.trainer.is_global_zero:
            # mesh = self.model.isosurface(can_space=False)
            # self.save_mesh(f"it{self.global_step}-{self.config.model.isosurface.resolution}.obj", mesh)
            res_path = self.get_save_path(f'results_it{self.global_step}-{prefix}.yaml')
            with open(res_path, 'w') as file:
                yaml.dump(metrics_dict, file)

            for _level in self.levels:
                idir = f"it{self.global_step}-{prefix}_{_level}"
                self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=30)

# from systems.utils import parse_optimizer, parse_scheduler
# @systems.register('instant_tnerf-system')
# class DNeRFSystem(TNeRFSystem):
    
#     def configure_optimizers(self):
#         optim = parse_optimizer(self.config.system.optimizer, self.model)
#         ret = {
#             'optimizer': optim,
#         }
#         if 'scheduler' in self.config.system:
#             ret.update({
#                 'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
#             })
#         return ret
