import torch
from nerfacc.data_specs import RayIntervals
from nerfacc.pdf import importance_sampling
from nerfacc.volrend import render_transmittance_from_density
from typing import Literal
from nerfacc.estimators.prop_net import PropNetEstimator

import models
from .base import BaseModel

class Sampler(BaseModel):
    def setup(self):
        self._stratified = self.config.get('stratified', True)
        self.sampling_type: Literal["uniform", "lindisp"] = self.config.get('sampling_type', 'uniform')

    @property
    def stratified(self):
        if self.training:
            return self._stratified
        return False

@models.register('uniform_sampler')
class UniformSampler(Sampler):
    def setup(self):
        super().setup()
        self._n_samples = self.config.n_samples
        self._n_importance = self.config.get('n_importance', 0)
        self.total_samples = self._n_samples + self._n_importance

    def forward(self, n_rays, near, far):
        """ Sample points along the ray.
        Args:
            rays_o: Tensor with shape (n_rays, 3).
            rays_d: Tensor with shape (n_rays, 3).
            near: Tensor with shape (n_rays).
            far: Tensor with shape (n_rays).
        Returns:
            points: Tensor with shape (n_rays, n_samples, 3).
            z_vals: Tensor with shape (n_rays, n_samples).
        """
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        intervals, _ = importance_sampling(
            intervals, cdfs, self.total_samples, self.stratified
        )
        t_vals = _transform_stot(
            self.sampling_type, intervals.vals, near.view(n_rays, 1), far.view(n_rays, 1)
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        return t_starts, t_ends

@models.register('importance_sampler')
class ImportanceSampler(Sampler):
    def setup(self):
        super().setup()
        self.n_samples = self.config.n_samples
        self.n_importance = self.config.n_importance
        if isinstance(self.n_samples, int):
            self.n_samples = [self.n_samples]

    @torch.no_grad()
    def forward(self, n_rays, near, far, prop_sigma_fns, requires_grad=False):
        """ Sample points along the ray.
        Args:
            rays_o: Tensor with shape (n_rays, 3).
            rays_d: Tensor with shape (n_rays, 3).
            near: Tensor with shape (n_rays).
            far: Tensor with shape (n_rays).
        Returns:
            A tuple of {Tensor, Tensor}:
            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).
        """
        is_list = isinstance(prop_sigma_fns, list)
        if is_list:
            assert len(prop_sigma_fns) == len(self.n_samples), (
                "The number of proposal networks and the number of samples "
                "should be the same."
            )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        # for level_fn, level_samples in zip(prop_sigma_fns, self.n_samples):
        for _level, level_samples in enumerate(self.n_samples):
            level_fn = prop_sigma_fns[_level] if is_list else prop_sigma_fns
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, self.stratified
            )
            t_vals = _transform_stot(
                self.sampling_type, intervals.vals, near, far
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas)
                cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)

        intervals, _ = importance_sampling(intervals, cdfs, self.n_importance, self.stratified)
        t_vals_fine = _transform_stot(
            self.sampling_type, intervals.vals, near, far
        )

        t_vals = torch.cat([t_vals, t_vals_fine], dim=-1)
        t_vals, _ = torch.sort(t_vals, dim=-1)

        t_starts_ = t_vals[..., :-1]
        t_ends_ = t_vals[..., 1:]

        return t_starts_, t_ends_

@models.register('proposal_sampler')
class ProposalSampler(Sampler, PropNetEstimator):
    def setup(self):
        super().setup()
        self.prop_samples = self.config.prop_samples
        self._n_samples = self.config.n_samples
        self._n_importance = self.config.get('n_importance', 0)
        self.total_samples = self._n_samples + self._n_importance
        if isinstance(self.prop_samples, int):
            self.prop_samples = [self.prop_samples]

    def forward(self, n_rays, near, far, prop_sigma_fns, requires_grad=False):
        t_starts_, t_ends_ = self.sampling(
            prop_sigma_fns=prop_sigma_fns,
            prop_samples=self.prop_samples,
            num_samples=self.total_samples,
            # rendering options
            n_rays=n_rays,
            near_plane=near,
            far_plane=far,
            sampling_type=self.sampling_type,
            # training options
            stratified=self.stratified,
            requires_grad=requires_grad,
        )
        return t_starts_, t_ends_
    
    def requires_grad_fn(self, target: float = 5.0, num_steps: int = 1000):
        # the proposal network is increasingly training less frequently until `num_steps`, 
        # when it levels out to training every `target` step
        schedule = lambda s: min(s / num_steps, 1.0) * target

        steps_since_last_grad = 0

        def proposal_requires_grad_fn(step: int) -> bool:
            nonlocal steps_since_last_grad
            target_steps_since_last_grad = schedule(step)
            requires_grad = steps_since_last_grad > target_steps_since_last_grad
            if requires_grad:
                steps_since_last_grad = 0
            steps_since_last_grad += 1
            return requires_grad

        return proposal_requires_grad_fn

def _transform_stot(
    transform_type: Literal["uniform", "lindisp"],
    s_vals: torch.Tensor, # n_rays, n_samples
    t_min: torch.Tensor,
    t_max: torch.Tensor,
) -> torch.Tensor:
    if transform_type == "uniform":
        _contract_fn, _icontract_fn = lambda x: x, lambda x: x
    elif transform_type == "lindisp":
        _contract_fn, _icontract_fn = lambda x: 1 / x, lambda x: 1 / x
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)
