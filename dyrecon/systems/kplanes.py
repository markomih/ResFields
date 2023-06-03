import math
import torch
import systems
from typing import Dict, Any

from collections import defaultdict, OrderedDict
from systems import criterions
from models.utils import masked_mean

from .tnerf import TNeRFSystem

from models.plenoxels.regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)

@systems.register('kplanes-system')
class KPlanesSystem(TNeRFSystem):
    
    # def configure_optimizers(self):
    #     optim_kwargs = self.config.system.optimizer

    #     optim_type = optim_kwargs['optim_type']
    #     if optim_type == 'adam':
    #         optim = torch.optim.Adam(params=self.model.get_params(optim_kwargs['lr']), eps=1e-15)
    #     else:
    #         raise NotImplementedError()

    #     ret = {
    #         'optimizer': optim,
    #     }
    #     import pdb; pdb.set_trace()

    #     if 'scheduler' in self.config.system:
    #         kwargs = self.config.system.scheduler
    #         eta_min = 0
    #         lr_sched = None
    #         max_steps = kwargs['num_steps']
    #         scheduler_type = kwargs['scheduler_type']
    #         if scheduler_type == "cosine":
    #             lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=eta_min)
    #         elif scheduler_type == "warmup_cosine":
    #             lr_sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=512, num_training_steps=max_steps)
    #         elif scheduler_type == "step":
    #             milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 5 // 6, max_steps * 9 // 10,]
    #             lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=0.33)
    #         elif scheduler_type == "warmup_step":
    #             milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 5 // 6, max_steps * 9 // 10,]
    #             lr_sched = get_step_schedule_with_warmup(optim, milestones, gamma=0.33, num_warmup_steps=512)
    #         else:
    #             raise NotImplementedError

    #         ret.update({
    #             'lr_scheduler': lr_sched,
    #         })

    #     return ret

    def prepare(self):
        super().prepare()

        self.ist_step = -1
        self.isg_step = -1
        self.compute_video_metrics = False
        self.regularizers = self.get_regularizers()


    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def get_regularizers(self):
        kwargs = self.config.system.loss
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    def _update_learning_rate(self):
        optimizer = self.optimizers()
        for g in optimizer.param_groups:
            lr = g['lr']
            return lr

    def _level_fn(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor]):
        loss, stats = 0, OrderedDict()
        loss_weight = self.config.system.loss
        stats["loss_rgb"] = masked_mean((out["rgb"] - batch["rgb"]) ** 2, batch["mask"].reshape(-1, 1))
        loss += stats["loss_rgb"]
        stats["metric_psnr"] = criterions.compute_psnr(out["rgb"], batch["rgb"], batch["mask"].reshape(-1, 1))

        # # TODO! consider adding back the mask loss
        # if 'mask' in batch:
        #     mask_loss = F.binary_cross_entropy(out['opacity'].clip(1e-3, 1.0 - 1e-3), (batch['mask']> 0.5).float())

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

        if self.training:
            for r in self.regularizers:
                if r.weight > 0.:
                    reg_loss = r.regularize(self.model, model_out=out)
                    stats[f"reg_{r.reg_type}"] = reg_loss.detach()
                    loss += reg_loss
        # if self.global_step % 10 == 0:
        #     print(self.model.field.grids[0][0].sum(), self.model.field.grids[0][1].sum())
        return loss, stats


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    """
    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_log_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 2e-5,
    eta_max: float = 1e-2,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))  # in [0,1]
        return math.exp(progress * math.log(eta_min) + (1 - progress) * math.log(eta_max)) / eta_max

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    milestones,
    gamma: float,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        out = 1.0
        for m in milestones:
            if current_step < m:
                break
            out *= gamma
        return out
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class EMA():
    def __init__(self, weighting=0.9):
        self.weighting = weighting
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.weighting * val + (1 - self.weighting) * self.val

    @property
    def value(self):
        return self.val

    def __str__(self):
        return f"{self.val:.2e}"
