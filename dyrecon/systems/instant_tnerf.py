from typing import Dict, Any
import torch
import systems
from typing import Dict, Any
from models.utils import masked_mean
from systems import criterions
from collections import OrderedDict
import torch
import torch.nn.functional as F

from .tnerf import TNeRFSystem

from systems.utils import parse_optimizer, parse_scheduler
from models.plenoxels.regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss, compute_plane_smoothness
)
@systems.register('instant_tnerf-system')
class InstantTNeRFSystem(TNeRFSystem):
    
    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })

        # if self.config.system.get('ema_decay', 0.0) > 0.0:
        #     ret.update({
        #         'ema': ExponentialMovingAverage(self.model.parameters(), decay=self.config.system.ema_decay)
        #     })
        return ret

    @torch.no_grad()
    def update_training_masks(self):
        n_frames = self.dataset.all_fg_masks.shape[0]
        error_map = self.model.error_images.unsqueeze(1)[:n_frames]  # n_frames, 1, H, W
        if True:
            threshold = error_map[error_map !=0.].mean()
        else:
            error_min, error_max = error_map.min(), error_map.max()
            error_range = error_max - error_min
            threshold = error_min + 0.6*error_range

        w_r = (error_map <= threshold).float()
        W_r = torch.nn.functional.avg_pool2d(w_r, (3, 3), padding=1, stride=1)  # B,1,H,W
        W_r = (W_r >= 0.5).float()
        w16 = torch.cat((W_r, torch.ones_like(W_r[..., :8])), dim=-1)
        w_R_8 = torch.nn.functional.avg_pool2d(w16, (16, 16))
        w_R_8 = torch.nn.Upsample(scale_factor=16, mode='nearest')(w_R_8)[..., :-8]
        w_R_8 = (w_R_8 >= 0.6).float()

        training_masks = w_R_8.clone().squeeze(1)
        # training_masks = W_r.clone().squeeze(1)
        return training_masks

    def vis_extra_images(self, batch, out):
        error_map = self.model.get_error_images(batch['time_index'])  # H, W
        median_error = self.model.error_images[self.model.error_images!=0.].mean()

        w_r = (error_map <= median_error).float()
        W_r = torch.nn.functional.avg_pool2d(w_r[None, None, ...], (3, 3), padding=1, stride=1)[0, 0]
        W_r = (W_r >= 0.5).float()
        w16 = torch.cat((W_r, torch.ones_like(W_r[:, :8])), dim=1)
        w_R_8 = torch.nn.functional.avg_pool2d(w16[None, None, ...], (16, 16))
        w_R_8 = torch.nn.Upsample(scale_factor=16, mode='nearest')(w_R_8)[0, 0][:, :-8]
        w_R_8 = (w_R_8 >= 0.6).float()

        # error_map_smooth = torch.nn.functional.avg_pool2d(error_map[None, None, ...], (3, 3), padding=1, stride=1)[0,0]
        # error_map_bin = (error_map_smooth > median_error).float()
        _log_img = [
            {'type': 'grayscale', 'img': error_map, 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': 1.0-w_r, 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': 1.0-W_r, 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
            {'type': 'grayscale', 'img': 1.0-w_R_8, 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
        ]
        # decompose images
        W, H = self.dataset.w, self.dataset.h
        if 'coarse' in out and 'w_xyz' in out['coarse']:
            _log_img = [ # w_xyz
                {'type': 'grayscale', 'img': out['coarse']['w_xyz'].reshape(H, W), 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
            ] + _log_img
        
        return _log_img

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor], level_name: str):
        loss, stats = super()._level_fn(batch, out, level_name)

        weight_w_xyz = self.config.system.loss.get('w_xyz', 0)
        if self.training and weight_w_xyz > 0:
            stats["loss_xyz_weights"] = ((1-out['xyz_weights'])**2).mean()
            loss += weight_w_xyz*stats["loss_depth"]

        weight_opacity = self.config.system.loss.get('mask_loss_bce', 0.0)
        if weight_opacity > 0.0:
            mask_loss_bce = F.binary_cross_entropy(
                out['opacity'].clip(1e-3, 1.0 - 1e-3).squeeze(),
                batch["mask"].squeeze())
            stats["mask_loss_bce"] = mask_loss_bce
            loss += weight_opacity*mask_loss_bce

        weight_sd_field_l1 = self.config.system.loss.get('sd_field_l1', 0.0)
        weight_sd_field_smoothness = self.config.system.loss.get('sd_field_smoothness', 0.0)
        if self.training and (weight_sd_field_l1 > 0.) or (weight_sd_field_smoothness > 0.):
            sd_field_l1_loss = 0
            sd_field_smoothness_loss = 0.0
            for grid_level in self.model.sd_comp_field.grids:
                for grid in grid_level:
                    sd_field_l1_loss += torch.abs(1 - grid).mean()
                    sd_field_smoothness_loss += compute_plane_smoothness(grid)

            stats["sd_field_l1"] = sd_field_l1_loss
            loss += weight_sd_field_l1*sd_field_l1_loss
        
        weight_static_pixel_masks = self.config.system.loss.get('weight_static_pixel_masks', 0.0)
        if self.training and weight_static_pixel_masks > 0.0:
            dynamic_pixel_masks = (batch['dynamic_pixel_masks'] > 0.).float()
            static_pixel_masks = 1.0 - dynamic_pixel_masks
            static_pixel_masks_loss = torch.nn.functional.binary_cross_entropy(out['w_xyz'].clip(1e-3, 1.0 - 1e-3).reshape(-1), static_pixel_masks)
            stats["static_pixel_masks_loss"] = static_pixel_masks_loss
            loss += weight_static_pixel_masks*static_pixel_masks_loss


        if self.training:
            fg_mask = batch["mask"].reshape(-1) > 0.0
            l1_error = torch.mean(torch.abs(out["rgb"] - batch["rgb"]), dim=-1)
            l1_error[~fg_mask] = 0
            self.model.update_error_images(batch['index'], batch['y'], batch['x'], l1_error)
            if self.global_step > 1 and (self.global_step % self.config.system.get('update_mask_every', 1e7)) == 0:
                training_masks = self.update_training_masks()
                self.dataset.all_fg_masks = self.dataset.all_fg_masks*training_masks

        return loss, stats

    def _update_learning_rate(self):
        optimizer = self.optimizers()
        for g in optimizer.param_groups:
            lr = g['lr']
            return lr


@systems.register('comp_instant_tnerf-system')
class CompInstantTNeRFSystem(TNeRFSystem):
    
    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret

    @torch.no_grad()
    def update_training_masks(self):
        n_frames = self.dataset.all_fg_masks.shape[0]
        error_map = self.model.error_images.unsqueeze(1)[:n_frames]  # n_frames, 1, H, W
        if True:
            threshold = error_map[error_map !=0.].mean()
        else:
            error_min, error_max = error_map.min(), error_map.max()
            error_range = error_max - error_min
            threshold = error_min + 0.6*error_range

        w_r = (error_map <= threshold).float()
        W_r = torch.nn.functional.avg_pool2d(w_r, (3, 3), padding=1, stride=1)  # B,1,H,W
        W_r = (W_r >= 0.5).float()
        w16 = torch.cat((W_r, torch.ones_like(W_r[..., :8])), dim=-1)
        w_R_8 = torch.nn.functional.avg_pool2d(w16, (16, 16))
        w_R_8 = torch.nn.Upsample(scale_factor=16, mode='nearest')(w_R_8)[..., :-8]
        w_R_8 = (w_R_8 >= 0.6).float()

        training_masks = w_R_8.clone().squeeze(1)
        # training_masks = W_r.clone().squeeze(1)
        return training_masks

    def vis_extra_images(self, batch, out):
        # decompose images
        W, H = self.dataset.w, self.dataset.h
        static_pixel_mask = (~(batch["dynamic_pixel_masks"].reshape(-1, 1)  > 0.)).float().reshape(H, W)
        _log_img = [ # w_xyz
            {'type': 'grayscale', 'img': static_pixel_mask, 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}},
        ]
        if out.get('w_xyz', None) is not None:
            _log_img.append(
                {'type': 'grayscale', 'img': out['w_xyz'].reshape(H, W)*out['opacity'].view(H, W), 'kwargs': {'cmap': 'jet', 'data_range': (0, 1)}}
            )
        
        return _log_img

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor], level_name: str):
        loss, stats = 0, OrderedDict()
        loss_weight = self.config.system.loss

        fg_mask = batch['mask'].reshape(-1, 1)
        dynamic_pixel_mask = (batch["dynamic_pixel_masks"].reshape(-1, 1)  > 0.).float()
        static_pixel_masks = 1.0 - dynamic_pixel_mask
        if level_name == 'static':
            fg_mask = fg_mask * static_pixel_masks
        elif level_name == 'dynamic':
            fg_mask = fg_mask * dynamic_pixel_mask
        elif level_name == 'blended':
            fg_mask = fg_mask
        else:
            raise NotImplementedError

        stats["loss_rgb"] = masked_mean((out["rgb"] - batch["rgb"]) ** 2, fg_mask)
        loss += stats["loss_rgb"]
        if 'depth' in batch:
            depth_mask = fg_mask.squeeze()*(batch["depth"] > 0).float().squeeze()
            stats["loss_depth"] = masked_mean(F.l1_loss(out["depth"], batch["depth"], reduction='none'), depth_mask)
            loss += loss_weight.depth*stats["loss_depth"]

        # logging
        stats["metric_psnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], fg_mask)
        if not self.training and level_name == 'blended':
            if 'covisible_masks' in batch:
                stats["metric_mpsnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["covisible_masks"].reshape(-1, 1))

            stats["metric_dynamic_mpsnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], dynamic_pixel_mask)

        # new loss
        weight_sd_field_l1 = loss_weight.get('sd_field_l1', 0.0)
        if self.training and (weight_sd_field_l1 > 0.):
            sd_field_l1_loss = 0
            # sd_field_smoothness_loss = 0.0
            for grid_level in self.model.sd_comp_field.grids:
                for grid in grid_level:
                    sd_field_l1_loss += torch.abs(grid).mean()
                    # sd_field_smoothness_loss += compute_plane_smoothness(grid)
            stats["sd_field_l1"] = sd_field_l1_loss
            loss += weight_sd_field_l1*sd_field_l1_loss

        weight_sparse_l1 = loss_weight.get('weight_sparse_l1', 0.0)
        if self.training and (weight_sparse_l1 > 0.):
            sparse_l1_loss = self.model.sd_comp_field.regularize()
            stats["sparse_l1"] = sparse_l1_loss
            loss += weight_sparse_l1*sparse_l1_loss

        weight_static_pixel_masks = loss_weight.get('weight_static_pixel_masks', 0.0)
        if level_name == 'static' and self.training and weight_static_pixel_masks > 0.0:
            stats["static_pixel_masks_loss"] = F.binary_cross_entropy(1.0-out['w_xyz'], static_pixel_masks.reshape(-1))
            loss += weight_static_pixel_masks*stats["static_pixel_masks_loss"]

        # if self.training:
        #     fg_mask = batch["mask"].reshape(-1) > 0.0
        #     l1_error = torch.mean(torch.abs(out["rgb"] - batch["rgb"]), dim=-1)
        #     l1_error[~fg_mask] = 0
        #     self.model.update_error_images(batch['index'], batch['y'], batch['x'], l1_error)
        #     if self.global_step > 1 and (self.global_step % self.config.system.get('update_mask_every', 1e7)) == 0:
        #         training_masks = self.update_training_masks()
        #         self.dataset.all_fg_masks = self.dataset.all_fg_masks*training_masks

        return loss, stats

    def _update_learning_rate(self):
        optimizer = self.optimizers()
        for g in optimizer.param_groups:
            lr = g['lr']
            return lr

    # def test_step(self, batch, batch_idx):  
    #     self.model.sd_comp_field.visualize_grid()
    #     return super().test_step(batch, batch_idx)

    def validation_epoch_end(self, out, prefix='val'):
        if self.trainer.is_global_zero:
            if getattr(self.model, 'sd_comp_field', None) is not None:
                dst_path = self.get_save_path(f'it{self.global_step}_{prefix}_compField')
                self.model.sd_comp_field.visualize_grid(dst_path)

        return super().validation_epoch_end(out, prefix)

# @systems.register('instant_tnerf-system-kplanes')
# class DNeRFSystem(TNeRFSystem):

#     def prepare(self):
#         super().prepare()
#         self.regularizers = self.get_regularizers()

#     def get_regularizers(self):
#         kwargs = self.config.system.loss
#         return [
#             PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
#             # PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
#             L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
#             # L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
#             TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
#             # TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
#             # HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
#             # DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
#         ]

#     def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor]):
#         loss, stats = super()._level_fn(batch, out)
#         if self.training:
#             for r in self.regularizers:
#                 if r.weight > 0.:
#                     reg_loss = r.regularize(self.model, model_out=out)
#                     stats[f"reg_{r.reg_type}"] = reg_loss.detach()
#                     loss += reg_loss
#         # if self.global_step % 10 == 0:
#         #     print(self.model.field.grids[0][0].sum(), self.model.field.grids[0][1].sum())
#         return loss, stats

#     def test_step(self, batch, batch_idx):
#         ret = super().test_step(batch, batch_idx)

#         if batch_idx == 0 and self.trainer.is_global_zero:
#             img_list = self.model.field.vis_time_planes()
#             for level, img in enumerate(img_list):
#                 img_name = f'timeplane_{level}'
#                 if self.logger is not None:
#                     self.logger.experiment.add_image(f'test/{img_name}', img/255., self.global_step, dataformats='HWC')
#                 # import cv2
#                 # cv2.imwrite(f'{img_name}.png', img)

#         return ret
