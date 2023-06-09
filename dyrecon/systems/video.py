import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import yaml
import skvideo.io
import systems
from typing import Any
from systems.base import BaseSystem
from utils import criterions
import torch.nn.functional as F

@systems.register('video_system')
class VideoSystem(BaseSystem):

    def prepare(self):
        if self.config.model.capacity == 'n_frames':
            self.config.model.capacity = self.config.model.metadata.n_frames
        self.sampling = self.config.model.sampling
        self.n_samples = self.sampling.n_samples // self.config.model.metadata.n_frames

    def forward(self, coords, frame_ids):
        # coords: (T, S, 3)
        # frame_ids: (T)
        # return: (T, S, 3))
        # split_size = batch_size // coords.shape[1]
        if not self.model.training: # batchify coords to prevent OOM at inference time
            pred = torch.cat([self.model(_c, _f) for _c, _f in zip(coords.split(1), frame_ids.split(1))], dim=0)
        else:
            pred = self.model(coords, frame_ids)
        # pred = torch.cat([self.model(_c, frame_ids) for _c in coords.split(split_size, dim=1)], dim=1)
        return pred
    
    def preprocess_data(self, batch, stage):
        coords = self.dataset.coords # T,H,W,3
        data = self.dataset.data # T,H,W,3

        n_frames = coords.shape[0]
        frame_ids = torch.arange(n_frames, device=coords.device)
        if stage in ['train']:
            t = frame_ids.unsqueeze(-1).repeat(1, self.n_samples).view(-1)
            y = torch.randint(0, coords.shape[1], size=(t.shape[0],), device=coords.device)
            x = torch.randint(0, coords.shape[2], size=(t.shape[0],), device=coords.device)
            coords = coords[t, y, x] # (F*S, 3)
            data = data[t, y, x]
        else:
            # add training coordinates to the batche to log the overfitting loss
            batch.update({
                'train_coords': self.dataset.train_coords.view(n_frames, -1, self.dataset.train_coords.shape[-1]),
                'train_data': self.dataset.train_data.view(n_frames, -1, self.dataset.train_data.shape[-1]),
            })
        batch.update({
            'coords': coords.view(n_frames, -1, coords.shape[-1]), # T,S,3
            'data': data.view(n_frames, -1, data.shape[-1]), # T,S,3
            'frame_ids': frame_ids, # T
        })

    def training_step(self, batch, batch_idx):
        pred = self(batch['coords'], batch['frame_ids'])
        loss = 1000.*F.mse_loss(pred, batch['data'])
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return dict(loss=loss)
    
    def validation_step(self, batch, batch_idx):
        metrics_dict = dict(metric_test_psnr=0, metric_train_psnr=0)
        if self.trainer.is_global_zero:
            test_pred = self(batch['coords'], batch['frame_ids'])
            test_psnr = criterions.compute_psnr(test_pred, batch['data']).mean()
            self.log('val/test_psnr', test_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
            # calculate overfitting loss
            if 'train_coords' in batch:
                train_pred = self(batch['train_coords'], batch['frame_ids'])
                train_psnr = criterions.compute_psnr(train_pred, batch['train_data']).mean()
                self.log('val/train_psnr', train_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
            
            res_path = self.get_save_path(f'results_it{self.global_step:06d}.yaml')
            metrics_dict = dict(metric_test_psnr=test_psnr.item(), metric_train_psnr=train_psnr.item())
            with open(res_path, 'w') as file:
                yaml.dump(metrics_dict, file)
        return metrics_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        if self.trainer.is_global_zero:
            test_pred = self(batch['coords'], batch['frame_ids'])
            gt_video = self.dataset.data
            test_pred = test_pred.view(self.dataset.data.shape)

            gt_video = (gt_video*255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)
            test_video = (test_pred*255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)

            # save video
            skvideo.io.vwrite(self.get_save_path(f'video_rnd_it{self.global_step:06d}.mp4'), test_video)
            skvideo.io.vwrite(self.get_save_path(f'video_gt.mp4'), gt_video)

            # log images
            for i in range(test_pred.shape[0]):
                fname = 'img%06d.png' % i
                rnd_path = self.get_save_path(os.path.join(f'img_rnd_it{self.global_step:06d}', fname))
                gt_path = self.get_save_path(os.path.join(f'img_gt', fname))

                cv2.imwrite(rnd_path, test_video[i,:,::-1])
                cv2.imwrite(gt_path, gt_video[i,:,::-1])
