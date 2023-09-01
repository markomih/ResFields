import numpy as np
import torch
import pytorch_lightning as pl
from utils.misc import get_rank

import models
from utils.mixins import SaverMixin
from utils.misc import config_to_primitive


class BaseSystem(pl.LightningModule, SaverMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = models.make(self.config.model.name, self.config.model)
        print(self.model)

    def prepare(self):
        pass

    def forward(self, batch):
        raise NotImplementedError
    
    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        return value
    
    def preprocess_data(self, batch, stage):
        pass

    def _get_metrics_dict(self, out, prefix):
        if self.trainer.is_global_zero and out != []:
            metrics = [k for k in out[0].keys() if 'loss' in k or 'metric' in k]
            metrics_dict = {}
            for key in metrics:
                metrics_dict[key] = float(torch.stack([step_out[key] for step_out in out]).mean().detach().cpu().item())
                self.log(f'{prefix}/{key}', metrics_dict[key], prog_bar=True, rank_zero_only=True, sync_dist=True)
                
            return metrics_dict

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    # on batch start
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')

    # on epoch start
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_predict_epoch_start(self):
        self.predict_step_outputs = []

    # on epoch end
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_step_outputs.append(outputs)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_step_outputs.append(outputs)

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.predict_step_outputs.append(outputs)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    # def validation_epoch_end(self, out):
    #     """
    #     Gather metrics from all devices, compute mean.
    #     Purge repeated results using data index.
    #     """
    #     raise NotImplementedError

    def test_step(self, batch, batch_idx):        
        raise NotImplementedError
    
    # def test_epoch_end(self, out):
    #     """
    #     Gather metrics from all devices, compute mean.
    #     Purge repeated results using data index.
    #     """
    #     raise NotImplementedError

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.config.system.optimizer.name)
        to_ret = {
            'optimizer':  optim(self.model.parameters(), **self.config.system.optimizer.args),
        }
        if self.config.system.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.config.system.scheduler.name)
            to_ret['lr_scheduler'] = {
                "scheduler": scheduler(to_ret['optimizer'], **self.config.system.scheduler.args),
                "interval": 'step', # The unit of the scheduler's step size
                "frequency": 1, # The frequency of the scheduler
            }
        return to_ret

    def _update_learning_rate(self):
        iter_step = self.global_step
        warm_up_end = self.config.system.scheduler.warm_up_end
        final_lr = self.config.system.scheduler.final_lr
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

        new_lr = learning_rate * learning_factor
        new_lr = max(new_lr, final_lr)
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        return new_lr
