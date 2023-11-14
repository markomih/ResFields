from functools import partial
import torch
import models
import torch.nn.functional as F
from nerfacc.volrend import render_weight_from_density, accumulate_along_rays
from models.base import BaseModel
from models.utils import chunk_batch_levels
from models.utils import extract_geometry, ray_bbox_intersection_nerfacc

@models.register('Renderer')
class Renderer(BaseModel):
    def setup(self):
        self.ray_chunk = self.config.sampling.ray_chunk
        self.background_color = None
        self.alpha_ratio = 1.0
        self.n_frames = self.config.metadata.n_frames
        self.sampler = models.make(self.config.sampling.name, self.config.sampling)
        if 'proposal' in self.config.sampling.name:
            # create prop net
            prop_net_cfg = self.config.sampling.prop_net
            self.prop_net = models.make(prop_net_cfg.model.name, prop_net_cfg.model)

        self.register_buffer('scene_aabb', torch.as_tensor(self.config.metadata.scene_aabb, dtype=torch.float32))
        self.config.dynamic_fields.metadata = self.config.metadata
        self.dynamic_fields = models.make(self.config.dynamic_fields.name, self.config.dynamic_fields)
        
        self.alpha_composing = self.config.get('alpha_composing', 'volsdf')
        assert self.alpha_composing in ['volsdf', 'neus', 'nerf'], 'Only support volsdf, neus, nerf'
        self.estimate_normals = self.alpha_composing in ['volsdf', 'neus']
        self.mc_threshold = 0.001
        if self.alpha_composing in ['volsdf', 'neus']:
            self.deviation_net = models.make(self.config.deviation_net.name, self.config.deviation_net)
            self.mc_threshold = 0
        self.vars_in_forward = {}

        self.fast_render = self.config.get('fast_render', False)
        if self.fast_render:
            self.config.grid_sampler.aabb = self.config.metadata.scene_aabb
            self.config.grid_sampler.n_frames = self.config.metadata.n_frames
            self.grid_sampler = models.make(self.config.grid_sampler.name, self.config.grid_sampler)

    def forward(self, rays, **kwargs):
        if self.training:
            out = self.forward_(rays, **kwargs)
        else:
            out = chunk_batch_levels(self.forward_, self.ray_chunk, rays, **kwargs)
        return {**out,}

    def forward_(self, rays, frame_id, **kwargs):
        rays_o, rays_d, rays_time = rays.split([3, 3, 1], dim=-1) #rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        # filter rays
        with torch.no_grad():
            near, far, mask = ray_bbox_intersection_nerfacc(self.scene_aabb.view(2, 3), rays_o, rays_d) # n_rays

        # prepare rendering dict
        to_ret = {
            'rgb': torch.zeros_like(rays_o),
            'opacity': torch.zeros_like(rays_o[:, :1]),
            'depth': torch.zeros_like(rays_o[:, :1]),
            'ray_mask': mask,
        }
        if self.estimate_normals:
            to_ret['normal'] = torch.zeros_like(rays_o)

        if mask.any():
            rays_o, rays_d, rays_time = rays_o[mask], rays_d[mask], rays_time[mask]
            if frame_id.numel() > 1:
                frame_id = frame_id[mask]
            if self.fast_render and not self.training and kwargs.get('fast_rendering', False):
                to_ret = self.volume_rendering_fast(rays_o, rays_d, rays_time, frame_id, near, far, to_ret)
            else:
                to_ret = self.volume_rendering(rays_o, rays_d, rays_time, frame_id, near, far, to_ret)
        else:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        return dict(coarse=to_ret)

    @torch.inference_mode()
    def cache_grid(self, frame_list, time_list, global_step):
        if self.fast_render and global_step > self.grid_sampler.cache_version:
            for frame_id, time_step in zip(frame_list, time_list):
                def query_density_fnc(pts):
                    out = self._query_field(pts, time_step, frame_id)
                    density = self._get_density(out)
                    return density
                self.grid_sampler.update(query_density_fnc, frame_id, global_step, occ_thre=0.0)

    def _query_field(self, pts, time_step, frame_id, fill_outside=False):
        assert time_step.numel() == 1 and frame_id.numel() == 1, 'Only support single time_step and frame_id'
        # pts: Tensor with shape (n_pts, 3). Dtype=float32.
        pts = pts.view(1, -1, 3).to(self.device)
        pts_time = torch.full_like(pts[..., :1], time_step)
        sdf = self.dynamic_fields(pts, pts_time, frame_id, None, alpha_ratio=self.alpha_ratio, estimate_normals=False, estimate_color=False)['sdf'].squeeze().clone()
        if fill_outside:
            fill_value = 0.0
            if self.alpha_composing in ['volsdf', 'neus']:
                fill_value = 1e4
            if self.scene_aabb is not None:
                eps = 0.025
                bmin = self.scene_aabb[:3][None] + eps
                bmax = self.scene_aabb[3:6][None] - eps
                inside_mask = (pts > bmin[None]).all(-1) & (pts < bmax[None]).all(-1)
                sdf[~inside_mask.squeeze()] = fill_value
            sdf = sdf.view(-1)
            if self.alpha_composing in ['volsdf', 'neus']:
                sdf = -sdf
        return sdf

    def isosurface(self, mesh_path, time_step, frame_id, resolution=None):
        assert time_step.numel() == 1 and frame_id.numel() == 1, 'Only support single time_step and frame_id'
        bound_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
        bound_max = torch.tensor([ 1.0,  1.0,  1.0], dtype=torch.float32)
        mesh = extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=self.mc_threshold,
            query_func=lambda _pts: self._query_field(_pts, time_step, frame_id, fill_outside=True))
        mesh.export(mesh_path)
        return mesh

    def regularizations(self, out):
        losses = super().regularizations(out)
        if self.training and 'proposal' in self.config.sampling.name:
            loss = self.sampler.compute_loss(self.vars_in_forward["trans"], loss_scaler=1.0)
            losses.update({'loss_proposal': loss})
            self.sampler.prop_cache = []
        return losses

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        if 'proposal' in self.config.sampling.name:
            self.vars_in_forward["requires_grad"] = self.sampler.requires_grad_fn(target=5, num_steps=global_step)

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if 'proposal' in self.config.sampling.name and self.training:
            self.sampler.prop_cache = []

    def sample_points(self, rays_o, rays_d, rays_time, frame_id, near, far):
        """ Render the volume with ray marching.
        Args:
            rays_o: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_d: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_time: Tensor with shape (n_rays, 1). Dtype=float32.
            frame_id: Tensor with shape (n_rays). Dtype=long.
            near: Tensor with shape (n_rays, 1). Dtype=float32.
            far: Tensor with shape (n_rays, 1). Dtype=float32.
        Returns:
            Start and end point on the ray (n_rays, num_samples):update
        """
        def _tdist2pts(tdist): # tdist: Tensor with shape (n_rays, n_samples) -> Tensor with shape (n_rays, n_samples, 3)
            _nr, _ns = tdist.shape[:2]
            return rays_o.view(_nr, 1, 3) + rays_d.view(_nr, 1, 3) * tdist.view(_nr, _ns, 1)

        n_rays = rays_o.shape[0]
        if 'uniform' in self.config.sampling.name:
            t_starts, t_ends = self.sampler(n_rays, near, far) # n_rays, n_samples
        elif 'importance' in self.config.sampling.name:
            def _prop_sigma_fns(_t_starts, _t_ends):
                _pts = _tdist2pts((_t_starts + _t_ends) * 0.5) #rays_o + rays_d * _pos
                _pts_time = rays_time.view(rays_time.shape[0], 1, 1).expand(-1, _pts.shape[1], -1)
                sdf = self.dynamic_fields(_pts, _pts_time, frame_id, None, alpha_ratio=self.alpha_ratio, estimate_normals=False, estimate_color=False)['sdf']
                density = self._get_density(sdf)
                return density.view(_pts.shape[:2])
            t_starts, t_ends = self.sampler(n_rays, near, far, prop_sigma_fns=_prop_sigma_fns) # n_rays, n_samples, 3
        elif 'proposal' in self.config.sampling.name:
            def _prop_sigma_fns(_t_starts, _t_ends, proposal_func):
                _pts = _tdist2pts((_t_starts + _t_ends) * 0.5) #rays_o + rays_d * _pos
                _pts_time = rays_time.view(rays_time.shape[0], 1, 1).expand(-1, _pts.shape[1], -1)
                sdf = proposal_func(_pts, _pts_time, frame_id, None, alpha_ratio=self.alpha_ratio, estimate_normals=False, estimate_color=False)['sdf']
                density = self._get_density(sdf)
                return density.view(_pts.shape[:2])
            # prop_sigma_fns = [partial(_prop_sigma_fns, proposal_func=_prop) for _prop in self.sampler.prop_nets]
            prop_sigma_fns = [partial(_prop_sigma_fns, proposal_func=self.prop_net)]
            # t_starts, t_ends = self.sampler(n_rays, near, far, prop_sigma_fns=prop_sigma_fns, requires_grad=self.vars_in_forward["requires_grad"]) # n_rays, n_samples, 3
            t_starts, t_ends = self.sampler(n_rays, near, far, prop_sigma_fns=prop_sigma_fns, requires_grad=self.vars_in_forward["requires_grad"]) # n_rays, n_samples, 3
        else:
            raise NotImplementedError

        return t_starts, t_ends

    def volume_rendering_fast(self, rays_o, rays_d, rays_time, frame_id, near, far, to_ret):
        """ Render the volume with ray marching.
        Args:
            rays_o: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_d: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_time: Tensor with shape (n_rays, 1). Dtype=float32.
            frame_id: Tensor with shape (n_rays). Dtype=long.
            near: Tensor with shape (n_rays). Dtype=float32.
            far: Tensor with shape (n_rays). Dtype=float32.
            to_ret: Dict of tensors.
        """
        assert frame_id.numel() == 1, 'Only support single frame_id'
        n_rays = rays_o.shape[0]
        ray_indices, t_starts, t_ends = self.grid_sampler(rays_o, rays_d, near, far, frame_id.item())
        if len(ray_indices) == 0:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])
            return to_ret

        t_mid = (t_starts + t_ends) * 0.5
        pts = rays_o[ray_indices] + t_mid[:, None]*rays_d[ray_indices] # [npts,3]

        # volume rendering
        out = self.dynamic_fields(
            pts.unsqueeze(0), # [1, npts, 3]
            rays_time[ray_indices].unsqueeze(0), # [1, npts, 1]
            frame_id, # (1,)
            rays_d[ray_indices].unsqueeze(0),
            alpha_ratio=self.alpha_ratio,
            estimate_normals=self.estimate_normals,
            estimate_color=True
        )
        out = {k: v.squeeze(0) if torch.is_tensor(v) else v for k, v in out.items()}
        sdf, color = out['sdf'], out['color']
        sigmas = self._get_density(sdf).squeeze(-1) # (npts,)
        weights = render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=n_rays)[0]

        to_ret['rgb'][to_ret['ray_mask']] = accumulate_along_rays(weights, color, ray_indices, n_rays) # n_rays, 3
        to_ret['opacity'][to_ret['ray_mask']] = accumulate_along_rays(weights, None, ray_indices, n_rays) # n_rays, 1
        to_ret['depth'][to_ret['ray_mask']] = accumulate_along_rays(weights, t_mid[:, None], ray_indices, n_rays) # n_rays, 1

        if self.background_color is not None:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        normal = out.get('normal', None)
        if self.estimate_normals and normal is not None:
            to_ret['normal'][to_ret['ray_mask']] = accumulate_along_rays(weights, normal, ray_indices, n_rays) # n_rays, 3

        return to_ret

    def volume_rendering(self, rays_o, rays_d, rays_time, frame_id, near, far, to_ret):
        """ Render the volume with ray marching.
        Args:
            rays_o: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_d: Tensor with shape (n_rays, 3). Dtype=float32.
            rays_time: Tensor with shape (n_rays, 1). Dtype=float32.
            frame_id: Tensor with shape (n_rays). Dtype=long.
            near: Tensor with shape (n_rays). Dtype=float32.
            far: Tensor with shape (n_rays). Dtype=float32.
            to_ret: Dict of tensors.
        """
        # pts, t_positions, t_ends, t_starts = self.sample_points(rays_o, rays_d, rays_time, frame_id, near.view(-1, 1), far.view(-1, 1))
        t_starts, t_ends = self.sample_points(rays_o, rays_d, rays_time, frame_id, near.view(-1, 1), far.view(-1, 1))
        t_mid = ((t_starts + t_ends) * 0.5).unsqueeze(-1)
        n_rays, n_samples = t_starts.shape
        # pts, z_vals = self.sampler(rays_o, rays_d, near, far) # n_rays, n_samples, 3
        pts_time = rays_time[:, None, :].expand(n_rays, n_samples, -1) # n_rays, n_samples, 1
        rays_d = rays_d[:, None, :].expand((n_rays, n_samples, -1)) # n_rays, n_samples, 3

        pts =  rays_o[:, None] + rays_d * t_mid # n_rays, n_samples, 3
        out = self.dynamic_fields(pts, pts_time, frame_id, rays_d, alpha_ratio=self.alpha_ratio, estimate_normals=self.estimate_normals, estimate_color=True)
        sdf, color, normal, gradients_o = out['sdf'], out['color'], out.get('normal', None), out.get('gradients_o', None)

        # volume rendering
        sigmas = self._get_density(sdf.squeeze(-1)) # n_rays, n_samples
        weights, trans, alphas = render_weight_from_density(t_starts, t_ends, sigmas)
        weights = weights.unsqueeze(-1) # n_rays, n_samples, 1
        
        to_ret['rgb'][to_ret['ray_mask']] = (color * weights).sum(dim=1) # n_rays, 3
        to_ret['opacity'][to_ret['ray_mask']] = weights.sum(dim=1) # n_rays, 1
        to_ret['depth'][to_ret['ray_mask']] = (weights*t_mid).sum(dim=1) # n_rays, 1

        if self.background_color is not None:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        if self.estimate_normals:
            to_ret['normal'][to_ret['ray_mask']] = (normal * weights).sum(dim=1) # n_rays, 3
            if self.training:
                to_ret['gradient_error'] = ((torch.linalg.norm(gradients_o, ord=2, dim=-1)-1.0)**2).mean()

        if self.training and 'proposal' in self.config.sampling.name:
            self.vars_in_forward["trans"] = trans.reshape(rays_o.shape[0], -1)

        if self.training:
            to_ret['sdf'] = sdf
        return to_ret

    def log_variables(self):
        to_ret = {}
        if self.alpha_composing in ['volsdf', 'neus']:
            to_ret['s_val'] = 1.0 / self.deviation_net.inv_s()
        return to_ret
    
    def _get_density(self, raw):
        """ Get density from raw out of a neural network.
        Args:
            raw: Tensor with shape (..., n_samples, 1). Dtype=float32.
        Returns:
            density: Tensor with shape (..., n_samples, 1). Dtype=float32.
        """
        if self.alpha_composing == 'volsdf':
            density = self.deviation_net(raw)
        elif self.alpha_composing == 'neus':
            raise NotImplementedError
        elif self.alpha_composing == 'nerf':
            density = F.relu(raw)
        else:
            raise NotImplementedError
        return density
