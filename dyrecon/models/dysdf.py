import torch
import models
from models.base import BaseModel

from models.utils import chunk_batch_levels
from nerfacc import ray_aabb_intersect
import torch.nn.functional as F

@models.register('DySDF')
class DySDF(BaseModel):
    def setup(self):
        # self.time_max = self.config.metadata.time_max
        self.n_samples = self.config.sampling.n_samples
        self.n_importance = self.config.sampling.n_importance
        self.randomized = self.config.sampling.randomized
        self.background_color = None
        self.alpha_ratio = 1.0
        
        self.register_buffer('scene_aabb', torch.as_tensor(self.config.metadata.scene_aabb, dtype=torch.float32))
        
        range = (self.scene_aabb[3:6] - self.scene_aabb[:3]).max().item()
        self.raw_noise_std = range/self.n_samples

        # create networks
        self.ambient_dim = self.config.get('ambient_dim', 0)
        self.deform_dim = self.config.get('deform_dim', 0) 

        self.ambient_codes = None
        if self.ambient_dim > 0:
            self.register_parameter('ambient_codes', torch.nn.Parameter(torch.randn(self.time_max+1, self.ambient_dim)))

        self.deform_codes = None
        if self.deform_dim > 0:
            self.register_parameter('deform_codes', torch.nn.Parameter(torch.randn(self.time_max+1, self.deform_dim)))

        self.deform_net = None
        if self.config.deform_net:
            self.deform_net = models.make(self.config.deform_net.name, self.config.hyper_net)

        self.hyper_net = None
        if self.config.hyper_net:
            self.hyper_net = models.make(self.config.hyper_net.name, self.config.hyper_net)
        self.hyper_codes = self.deform_codes

        self.sdf_net = models.make(self.config.sdf_net.name, self.config.sdf_net)
        self.color_net = models.make(self.config.color_net.name, self.config.color_net)
        self.deviation_net = models.make(self.config.deviation_net.name, self.config.deviation_net)

    def train(self, mode=True):
        self.randomized = mode and self.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()

    def forward(self, rays, **kwargs):
        if self.training:
            out = self.forward_(rays, **kwargs)
        else:
            out = chunk_batch_levels(self.forward_, self.config.ray_chunk, rays, **kwargs)
        return {
            **out,
        }

    def forward_(self, rays, rays_time, frame_id, **kwargs):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        renderings = self.volume_rendering(rays_o, rays_d, rays_time, frame_id)
        return dict(coarse=renderings)

    def isosurface(self, resolution=None):
        return

    def get_render_step_size(self):
        """ Get the render step size based on the size of bounding box."""
        bbsize = (self.scene_aabb[3:6] - self.scene_aabb[0:3]).max().item()
        return bbsize/self.n_samples

    def volume_rendering(self, rays_o, rays_d, rays_time, frame_id):
        """ Render the volume with ray marching.
        Args:
            rays_o: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_d: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_time: Tensor with shape (n_rays, 1). Dtype=float32.
            frame_id: Tensor with shape (n_rays). Dtype=long.
        """
        to_ret = {
            'rgb': torch.zeros_like(rays_o),
            'opacity': torch.zeros_like(rays_o[:, :1]),
            'normal_map': torch.zeros_like(rays_o),
            'depth_map': torch.zeros_like(rays_o[:, :1]),
        }

        # sample points for volume rendering
        render_step_size = self.get_render_step_size()
        with torch.no_grad():
            near, far = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb) # n_rays
            print('FAR?MIN', (near.min(), near.max()), (far.min(), far.max()))
            inside_rays = near != far
            if not inside_rays.any():
                to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])
                return to_ret
            rays_o, rays_d, rays_time, frame_id, near, far = rays_o[inside_rays], rays_d[inside_rays], rays_time[inside_rays], frame_id[inside_rays], near[inside_rays], far[inside_rays]

            z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device) # n_rays
            z_vals = near.view(-1, 1) + (far - near).view(-1, 1) * z_vals.unsqueeze(0) # n_rays, n_samples

            if self.randomized: # stratified
                t_rand = torch.rand_like(z_vals) - 0.5
                z_vals = z_vals + t_rand * render_step_size

            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.tensor([render_step_size], device=dists.device).expand(dists[..., :1].shape)], -1) # n_rays, n_samples
            mid_z_vals = z_vals + dists * 0.5
    
            pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
            rays_time = rays_time[:, None, :].expand(-1, pts.shape[1], -1) # n_rays, n_samples, 1
            rays_d = rays_d[:, None, :].expand(pts.shape) # n_rays, n_samples, 3

        # Forward through networks
        with torch.inference_mode(False), torch.enable_grad():  # enable gradient for computing gradients
            if not self.training:
                pts = pts.clone()

            pts.requires_grad_(True)

            deform_codes = self.deform_codes[frame_id].unsqueeze(1).expand(-1, pts.shape[1], -1) if self.deform_codes is not None else None
            ambient_codes = self.ambient_codes[frame_id].unsqueeze(1).expand(-1, pts.shape[1], -1) if self.ambient_codes is not None else None
            hyper_codes = self.hyper_codes[frame_id].unsqueeze(1).expand(-1, pts.shape[1], -1) if self.hyper_codes is not None else None

            pts_canonical = pts if self.deform_net is None else self.deform_net(deform_codes, pts, self.alpha_ratio, rays_time)
            hyper_coord = rays_time if self.hyper_net is None else self.hyper_net(hyper_codes, pts, rays_time, self.alpha_ratio)
                
            sdf_nn_output = self.sdf_net(pts_canonical, hyper_coord, self.alpha_ratio, frame_id=frame_id)
            sdf, feature_vector = sdf_nn_output[..., :1], sdf_nn_output[..., 1:] # (n_rays, n_samples, 1), (n_rays, n_samples, F)

            gradients_o =  torch.autograd.grad(outputs=sdf, inputs=pts, grad_outputs=torch.ones_like(sdf, requires_grad=False, device=sdf.device), create_graph=True, retain_graph=True, only_inputs=True)[0]

        color = self.color_net(feature=feature_vector, point=pts_canonical, ambient_code=ambient_codes, view_dir=rays_d, normal=gradients_o, alpha_ratio=self.alpha_ratio) # n_rays, n_samples, 3

        # volume rendering
        weights = self.get_weight_vol_sdf(sdf, dists) # n_rays, n_samples, 1
        
        comp_rgb = (color * weights).sum(dim=1) # n_rays, 3
        opacity = weights.sum(dim=1) # n_rays, 1
        depth_map = (weights*mid_z_vals.unsqueeze(-1)).sum(dim=1) # n_rays, 1
        comp_normal = (F.normalize(gradients_o, dim=-1, p=2) * weights).sum(dim=1) # n_rays, 3

        to_ret['rgb'][inside_rays] = comp_rgb
        to_ret['opacity'][inside_rays] = opacity
        to_ret['normal_map'][inside_rays] = comp_normal
        to_ret['depth_map'][inside_rays] = depth_map

        if self.background_color is not None:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        if self.training:
            to_ret['gradient_error'] = ((torch.linalg.norm(gradients_o, ord=2, dim=-1)-1.0)**2).mean()
        return to_ret

    def log_variables(self):
        return {
            's_val': 1.0 / self.deviation_net.inv_s(),
        }
    
    def get_weight_vol_sdf(self, sdf, dists):
        """ Compute the weights for volume rendering with the formulation from VolSDF.
        Args:
            sdf: Tensor with shape (n_rays, n_samples, 1). Dtype=float32.
            dists: Tensor with shape (n_rays, n_samples). Dtype=float32.
        Returns:
            weights: Tensor with shape (n_rays, n_samples). Dtype=float32.
        """
        density = self.deviation_net(sdf)
        alpha = 1.0 - torch.exp(-density * dists.view(density.shape))
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-7], dim=1), dim=1)[:, :-1] # n_rays, n_samples, 1
        weights = alpha * transmittance
        return weights
