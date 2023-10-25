import torch

import models
from .base import BaseModel

class Sampler(BaseModel):
    def setup(self):
        return super().setup()
    
    def compute_z_vals(self, **kwargs):
        raise NotImplementedError

    def compute_t_vals(self, **kwargs):
        raise NotImplementedError

@models.register('uniform_sampler')
class UniformSampler(Sampler):
    def setup(self):
        self.n_samples = self.config.n_samples
        self._stratified = self.config.get('stratified', True)

    @property
    def stratified(self):
        if self.training:
            return self._stratified
        return False

    def compute_t_vals(self, device):
        t_vals = torch.linspace(0., 1., steps=self.n_samples+1, device=device)
        if self.stratified:
            interval = 1/self.n_samples
            t_vals = torch.rand_like(t_vals) * interval + t_vals 
        return t_vals
    
    def compute_z_vals(self, t_vals, near, far):
        """ Sample points along the ray.
        Args:
            t_bals: Tensor with shape (n_samples).
            near: Tensor with shape (n_rays).
            far: Tensor with shape (n_rays).
        """
        # near, far = rays[..., 6:7], rays[..., 7:8]
        z_vals = near.view(-1, 1) + (far - near).view(-1, 1) * t_vals.view(1, -1) # n_rays, n_samples
        return z_vals
        
    def forward(self, rays_o, rays_d, near, far):
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
        t_vals = self.compute_t_vals(rays_o.device)
        z_vals = self.compute_z_vals(t_vals, near, far)
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_mid[..., :, None]
        return points, z_vals
    