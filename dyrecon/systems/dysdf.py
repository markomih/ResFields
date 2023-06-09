import os
import torch
import torch.nn.functional as F
import yaml
import systems
from typing import Dict, Any
from collections import OrderedDict
from systems.base import BaseSystem
from models.utils import masked_mean
from utils import criterions

@systems.register('dysdf_system')
class DySDFSystem(BaseSystem):
    def prepare(self):
        self.sampling = self.config.model.sampling
        self.train_num_rays = self.sampling.train_num_rays

    def forward(self, batch):
        return self.model(**batch) 
    
    def preprocess_data(self, batch, stage):
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch[key] = val.squeeze(0).to(self.device)
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
        if 'rgb' in batch:
            batch['rgb'] = batch['rgb']*batch['mask'][..., None] + self.model.background_color * (1 - batch['mask'][..., None])
        setattr(self.model, 'alpha_ratio', self.alpha_ratio)

    @property
    def alpha_ratio(self):
        alpha_ratio_cfg = self.config.model.get('alpha_ratio', None)
        alpha_ratio = 1.0
        if alpha_ratio_cfg:
            strategy = alpha_ratio_cfg.get('strategy', 'interpolated')
            if strategy == 'interpolated':
                alpha_ratio = max(min(self.global_step/alpha_ratio_cfg.get('max_steps', 50000), 1.), 0.)
        return alpha_ratio

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor], level_name: str):
        loss, stats = 0, OrderedDict()
        loss_weight = self.config.system.loss
        if 'rgb' in batch and loss_weight.get('rgb', 0.0) > 0.0:
            stats['loss_rgb'] = masked_mean((out["rgb"] - batch["rgb"]).abs(), batch["mask"].reshape(-1, 1))
            loss += loss_weight.rgb*stats['loss_rgb']

        if 'mask' in batch and loss_weight.get('mask', 0.0) > 0.0:
            stats['loss_mask'] = F.binary_cross_entropy(out['opacity'].clip(1e-3, 1.0 - 1e-3).squeeze(), (batch['mask']> 0.5).float().squeeze())
            loss += loss_weight.mask*stats['loss_mask']

        if 'gradient_error' in out and loss_weight.get('eikonal', 0.0) > 0.0:
            stats["loss_eikonal"] = out["gradient_error"]
            loss += loss_weight.eikonal * stats["loss_eikonal"]

        # compute test metrics
        if not self.training:
            W, H = self.dataset.w, self.dataset.h
            stats.update(dict(
                metric_mpsnr = criterions.compute_psnr(out['rgb'], batch['rgb'], batch['mask'].view(-1, 1)),
                metric_psnr = criterions.compute_psnr(out['rgb'], batch['rgb']),
                metric_ssim = criterions.compute_ssim(out['rgb'].view(H, W, 3), batch['rgb'].view(H, W, 3)),
                metric_mask_bce = F.binary_cross_entropy((batch['mask'].squeeze()).float(), out['opacity'].squeeze()),
            ))

        return loss, stats

    def training_step(self, batch, batch_idx):
        out = self(batch)
        # regularizations = self.model.regularization_loss(batch)

        losses = dict()
        self.levels = out.keys()
        for _level in self.levels:
            losses[_level], stats = self._level_fn(batch, out[_level], _level)
            for key, val in stats.items():
                self.log(f'train/{_level}_{key}', val, prog_bar=False, rank_zero_only=True, sync_dist=True)

        loss = sum(losses.values())
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        _log_variables = self.model.log_variables()
        for key, val in _log_variables.items():
            self.log(f'train/{key}', val, prog_bar=False, rank_zero_only=True, sync_dist=True)

        return {
            'loss': loss
        }

    def vis_extra_images(self, batch, out):
        return []

    def validation_step(self, batch, batch_idx, prefix='val'):
        out = self(batch) #rays (N, 7), rgb (N,3), depth (N,1)
        W, H = self.dataset.w, self.dataset.h

        stats_dict = dict()
        self.levels = out.keys()
        for _level in self.levels:
            loss, stats = self._level_fn(batch, out[_level], _level)

            stats_dict.update({f"{_level}_{key}": val for key, val in stats.items()})
            stats_dict[f'{_level}_loss'] = loss
            
            _log_imgs = [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out[_level]['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ]
            file_name = f"{batch['index'].squeeze().item():06d}"
            if prefix == 'test':
                self.save_image_grid(f"rgb_gt/it{self.global_step:06d}-{prefix}_{_level}/{file_name}.png", [_log_imgs[-2]])
                self.save_image_grid(f"rgb/it{self.global_step:06d}-{prefix}_{_level}/{file_name}.png", [_log_imgs[-1]])
            _log_imgs += self.vis_extra_images(batch, out[_level])
            if 'normal' in out[_level]:
                normal = (0.5 + 0.5*out[_level]['normal'].view(H, W, 3))*out[_level]['opacity'].view(H, W, 1)
                normal = normal + self.model.background_color.view(1, 1, 3).expand(normal.shape)*(1.0 - out[_level]['opacity'].view(H, W, 1))
                _log_imgs.append(
                    {'type': 'rgb', 'img': normal, 'kwargs': {'data_format': 'HWC'}},
                )
                if prefix == 'test':
                    self.save_image_grid(f"normal/it{self.global_step:06d}-{prefix}_{_level}/{file_name}.png", [_log_imgs[-1]])
            if 'depth' in out[_level]:
                _log_imgs += [
                    {'type': 'grayscale', 'img': out[_level]['depth'].view(H, W), 'kwargs': {}},
                ]

            _log_imgs += [
                {'type': 'grayscale', 'img': out[_level]['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            ]

            img = self.save_image_grid(f"it{self.global_step:06d}-{prefix}_{_level}/{file_name}.png", _log_imgs)
            if self.trainer.is_global_zero:
                if self.logger is not None:
                    if 'WandbLogger' in str(type(self.logger)):
                        self.logger.log_image(key=f'{prefix}/{_level}_renderings', images=[img/255.], caption=["renderings"])
                    else:
                        self.logger.experiment.add_image(f'{prefix}/{_level}_renderings', img/255., self.global_step, dataformats='HWC')


        stats_dict['index'] = batch['index']
        return stats_dict

    def _get_metrics_dict(self, out, prefix):
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
                self.log(f'{prefix}/{key}', m_val, prog_bar=True, rank_zero_only=True, sync_dist=True)
            return metrics_dict

    def on_validation_epoch_end(self, prefix='val'):
        out = self.all_gather(self.validation_step_outputs)
        if self.trainer.is_global_zero:
            metrics_dict = self._get_metrics_dict(out, prefix)
            return metrics_dict

    def test_step(self, batch, batch_idx):
        frame_id = batch['frame_id']
        file_name = f"{int(frame_id.item()):06d}"
        mesh_path = self.get_save_path(f"meshes/it{self.global_step:06d}/{file_name}.ply")
        if not os.path.exists(mesh_path):
            time_step = self.dataset.frame_id_to_time(frame_id)
            self.model.isosurface(mesh_path, time_step, frame_id, self.config.model.isosurface.resolution)
        return self.validation_step(batch, batch_idx, prefix='test')

    # def on_test_epoch_end(self, out):
    def on_test_epoch_end(self, prefix='test'):
        out = self.all_gather(self.test_step_outputs)
        if self.trainer.is_global_zero:
            metrics_dict = self._get_metrics_dict(out, prefix)
            res_path = self.get_save_path(f'results_it{self.global_step:06d}-{prefix}.yaml')
            with open(res_path, 'w') as file:
                yaml.dump(metrics_dict, file)

            for _level in self.levels:
                idir = f"it{self.global_step:06d}-{prefix}_{_level}"
                self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=30)
