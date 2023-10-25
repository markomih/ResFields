import torch
import models
import torch.nn.functional as F

from models.base import BaseModel
from models.utils import chunk_batch_levels
from models.utils import ray_bbox_intersection, extract_geometry, ray_bbox_intersection_nerfacc

@models.register('DySDF')
class DySDF(BaseModel):
    def setup(self):
        # self.n_importance = self.config.sampling.n_importance
        self.ray_chunk = self.config.sampling.ray_chunk
        self.background_color = None
        self.alpha_ratio = 1.0
        self.n_frames = self.config.metadata.n_frames
        self.sampler = models.make(self.config.sampling.name, self.config.sampling)
        
        self.register_buffer('scene_aabb', torch.as_tensor(self.config.metadata.scene_aabb, dtype=torch.float32))
        

        # create networks
        self.ambient_dim = self.config.get('ambient_dim', 0)
        self.deform_dim = self.config.get('deform_dim', 0) 

        if self.ambient_dim > 0:
            self.register_parameter('ambient_codes', torch.nn.Parameter(torch.randn(self.n_frames, self.ambient_dim)))
        else:
            self.ambient_codes = None

        if self.deform_dim > 0:
            self.deform_codes = torch.nn.Parameter(torch.randn(self.n_frames, self.deform_dim))
        else:
            self.deform_codes = None

        if self.config.deform_net:
            self.deform_net = models.make(self.config.deform_net.name, self.config.deform_net)
        else:
            self.deform_net = None

        self.hyper_net = models.make(self.config.hyper_net.name, self.config.hyper_net)
        self.config.sdf_net.d_in_2 = self.hyper_net.out_dim
        self.config.sdf_net.n_frames = self.n_frames
        self.sdf_net = models.make(self.config.sdf_net.name, self.config.sdf_net)
        self.color_net = models.make(self.config.color_net.name, self.config.color_net)
        
        self.alpha_composing = self.config.get('alpha_composing', 'volsdf')
        assert self.alpha_composing in ['volsdf', 'neus', 'nerf'], 'Only support volsdf, neus, nerf'
        self.estimate_normals = self.alpha_composing in ['volsdf', 'neus']
        if self.alpha_composing in ['volsdf', 'neus']:
            self.deviation_net = models.make(self.config.deviation_net.name, self.config.deviation_net)
            self.mc_threshold = 0
        else:
            self.mc_threshold = 0.001

    def forward(self, rays, **kwargs):
        if self.training:
            out = self.forward_(rays, **kwargs)
        else:
            out = chunk_batch_levels(self.forward_, self.ray_chunk, rays, **kwargs)
        return {
            **out,
        }

    def filter_rays(self, rays_o, rays_d, rays_time, frame_id):
        near, far, mask = ray_bbox_intersection_nerfacc(self.scene_aabb.view(2, 3), rays_o, rays_d) # n_rays
        rays_o, rays_d, rays_time = rays_o[mask], rays_d[mask], rays_time[mask]
        if frame_id.numel() > 1:
            frame_id = frame_id[mask]
        return rays_o, rays_d, rays_time, frame_id, mask, near, far

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

            to_ret = self.volume_rendering(rays_o, rays_d, rays_time, frame_id, near, far, to_ret)
        else:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        return dict(coarse=to_ret)

    def _query_sdf(self, pts, frame_id, time_step):
        # pts: Tensor with shape (B, n_pts, 3). Dtype=float32.
        # frame_id: Tensor with shape (B). Dtype=long.
        # time_step: Tensor with shape (B,1). Dtype=float32.
        _deform_codes = self.deform_codes[frame_id] if self.deform_codes is not None else None
        if _deform_codes is not None:
            deform_codes = _deform_codes.view(-1, 1, _deform_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)
        else: 
            deform_codes = _deform_codes

        pts_time = time_step[:, None, :].expand(-1, pts.shape[1], -1)
        pts_canonical = pts if self.deform_net is None else self.deform_net(deform_codes, pts, self.alpha_ratio, pts_time)
        hyper_coord = self.hyper_net(deform_codes, pts, pts_time, self.alpha_ratio)
            
        sdf = self.sdf_net(pts_canonical, hyper_coord, self.alpha_ratio, input_time=time_step, frame_id=frame_id)[..., :1]
        sdf = sdf.squeeze()
        return sdf

    def isosurface(self, mesh_path, time_step, frame_id, resolution=None):
        assert time_step.numel() == 1 and frame_id.numel() == 1, 'Only support single time_step and frame_id'
        _deform_codes = self.deform_codes[frame_id] if self.deform_codes is not None else None

        def _query_sdf(pts):
            # pts: Tensor with shape (n_pts, 3). Dtype=float32.
            pts = pts.view(1, -1, 3).to(frame_id.device)
            deform_codes = None
            if _deform_codes is not None:
                deform_codes = _deform_codes.view(-1, 1, _deform_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)

            pts_time = torch.full_like(pts[..., :1], time_step)

            pts_canonical = pts if self.deform_net is None else self.deform_net(deform_codes, pts, self.alpha_ratio, pts_time)
            hyper_coord = self.hyper_net(deform_codes, pts, pts_time, self.alpha_ratio)
                
            sdf = self.sdf_net(pts_canonical, hyper_coord, self.alpha_ratio, input_time=time_step, frame_id=frame_id)[..., :1]
            sdf = sdf.squeeze()

            # sdf = self.sdf_network.sdf(pts_canonical, ambient_coord, self.alpha_ratio, frame_id=time_step)
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
        
        bound_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
        bound_max = torch.tensor([ 1.0,  1.0,  1.0], dtype=torch.float32)
        mesh = extract_geometry(bound_min, bound_max, resolution=resolution, threshold=self.mc_threshold, query_func=_query_sdf)
        mesh.export(mesh_path)
        return mesh

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
        pts, z_vals = self.sampler(rays_o, rays_d, near, far) # n_rays, n_samples, 3
        pts_time = rays_time[:, None, :].expand(-1, pts.shape[1], -1) # n_rays, n_samples, 1
        rays_d = rays_d[:, None, :].expand(pts.shape) # n_rays, n_samples, 3
        if frame_id.numel() == 1:
            rays_time = rays_time.view(-1)[0]

        # Forward through networks
        with torch.inference_mode(False), torch.enable_grad():  # enable gradient for computing gradients
            if not self.training:
                pts = pts.clone()

            pts.requires_grad_(True)
            deform_codes = self.deform_codes[frame_id] if self.deform_codes is not None else None
            if deform_codes is not None:
                deform_codes = deform_codes.view(-1, 1, deform_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)
            ambient_codes = self.ambient_codes[frame_id] if self.ambient_codes is not None else None
            if ambient_codes is not None:
                ambient_codes = ambient_codes.view(-1, 1, ambient_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)

            pts_canonical = pts if self.deform_net is None else self.deform_net(deform_codes, pts, self.alpha_ratio, pts_time)
            hyper_coord = self.hyper_net(deform_codes, pts, pts_time, self.alpha_ratio)

            sdf_nn_output = self.sdf_net(pts_canonical, hyper_coord, self.alpha_ratio, input_time=rays_time.squeeze(-1), frame_id=frame_id.squeeze(-1))
            sdf, feature_vector = sdf_nn_output[..., :1], sdf_nn_output[..., 1:] # (n_rays, n_samples, 1), (n_rays, n_samples, F)

            if self.estimate_normals:
                gradients_o =  torch.autograd.grad(outputs=sdf, inputs=pts, grad_outputs=torch.ones_like(sdf, requires_grad=False, device=sdf.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients_o = None

        color = self.color_net(feature=feature_vector, point=pts_canonical, ambient_code=ambient_codes, view_dir=rays_d, normal=gradients_o, alpha_ratio=self.alpha_ratio) # n_rays, n_samples, 3

        # volume rendering
        weights = self.get_weight(sdf, z_vals) # n_rays, n_samples, 1
        
        to_ret['rgb'][to_ret['ray_mask']] = (color * weights).sum(dim=1) # n_rays, 3
        to_ret['opacity'][to_ret['ray_mask']] = weights.sum(dim=1) # n_rays, 1
        to_ret['depth'][to_ret['ray_mask']] = (weights*(.5 * (z_vals[..., 1:] + z_vals[..., :-1])).unsqueeze(-1)).sum(dim=1) # n_rays, 1

        if self.background_color is not None:
            to_ret['rgb'] = to_ret['rgb'] + self.background_color.view(-1, 3).expand(to_ret['rgb'].shape)*(1.0 - to_ret['opacity'])

        if self.estimate_normals:
            to_ret['normal'][to_ret['ray_mask']] = (F.normalize(gradients_o, dim=-1, p=2) * weights).sum(dim=1) # n_rays, 3
            if self.training:
                to_ret['gradient_error'] = ((torch.linalg.norm(gradients_o, ord=2, dim=-1)-1.0)**2).mean()
        if self.training:
            to_ret['sdf'] = sdf
        return to_ret

    def log_variables(self):
        to_ret = {}
        if self.alpha_composing in ['volsdf', 'neus']:
            to_ret['s_val'] = 1.0 / self.deviation_net.inv_s()
        return to_ret
    
    def get_weight(self, nn_output, z_vals):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        if self.alpha_composing == 'volsdf':
            weight = self._get_weight_vol_sdf(nn_output, dists)
        elif self.alpha_composing == 'neus':
            raise NotImplementedError
        elif self.alpha_composing == 'nerf':
            weight = self._get_weight_nerf(nn_output, dists)
        else:
            raise NotImplementedError
        return weight

    def _get_weight_vol_sdf(self, sdf, dists):
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

    def _get_weight_nerf(self, raw, dists):
        density = F.relu(raw)
        alpha = 1.0 - torch.exp(-density * dists.view(density.shape))
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-7], dim=1), dim=1)[:, :-1] # n_rays, n_samples, 1
        weights = alpha * transmittance
        return weights
