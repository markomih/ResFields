import torch
import models
from models.base import BaseModel

from models.nerf_utils import NerfMLP, SinusoidalEncoder, MLP, VarianceNetwork, raw2outputs, sample_pdf, resample_zdist, DNeRFWarp, HyperNeRFAmbientNetwork, SE3FieldNerfies
from models.utils import chunk_batch_levels

class AmbientNeRFMLP(torch.nn.Module):
    def __init__(
        self,
        x_in: int = 3,  # input point dimensions
        ambient_dim: int = 0,  # input time dimensions
        hyper_dim: int = 0,  # input hyper dimensions
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        independent_layers: list = [],
        composition_rank: int = 10,
        n_frames: int = 1000,
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(x_in, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        if hyper_dim > 0:
            self.hyper_encoder = SinusoidalEncoder(hyper_dim, 0, 10, True)
            hyper_dim = self.hyper_encoder.latent_dim
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim + ambient_dim + hyper_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            independent_layers = independent_layers,
            composition_rank = composition_rank,
            n_frames = n_frames,
        )

    def forward(self, x, topo_code=None, ambient_code=None, rays_d=None, frame_id=None):
        inputs = [self.posi_encoder(x)]
        if topo_code is not None:
            inputs.append(self.hyper_encoder(topo_code))
        if ambient_code is not None:
            inputs.append(ambient_code)
        inputs = torch.cat(inputs, dim=-1)
    
        if rays_d is not None:
            rays_d = self.view_encoder(rays_d)
        rgb, sigma = self.mlp(inputs, condition=rays_d, frame_id=frame_id)
        return torch.cat((rgb, sigma), dim=-1)

class AmbientNeuSMLP(torch.nn.Module):
    def __init__(
        self,
        x_in: int = 3,  # input point dimensions
        ambient_dim: int = 0,  # input time dimensions
        hyper_dim: int = 0,  # input hyper dimensions
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        # 2nd mlp parameters
        color_net_depth: int = 5,
        color_net_width: int = 256,
    ):
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(x_in, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.cos_anneal_ratio = 1.0
        if hyper_dim > 0:
            self.hyper_encoder = SinusoidalEncoder(hyper_dim, 0, 10, True)
            hyper_dim = self.hyper_encoder.latent_dim

        self.variance = VarianceNetwork() #deviation_network

        self.sdf_network = MLP(
            input_dim=self.posi_encoder.latent_dim + ambient_dim + hyper_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=True,
            output_dim=net_width,
            hidden_activation=torch.nn.Softplus(beta=100.)
        )
        hidden_features = self.sdf_network.output_dim
        self.color_network = MLP(
            input_dim=3 + 3 + 3 + (hidden_features-1) + ambient_dim,  #condition_dim is ambient dim
            net_depth=color_net_depth,
            net_width=color_net_width,
            output_enabled=True,
            output_dim=3,
            hidden_activation=torch.relu
        )

    def forward(self, points_can, rays_d_can, topo_code=None, ambient_code=None):
        with torch.inference_mode(False), torch.enable_grad():
            if not self.training:
                points_can = points_can.clone()
            points_can.requires_grad_(True)
            # compute normals
            sdf_inputs = [self.posi_encoder(points_can)]
            if topo_code is not None:
                sdf_inputs.append(self.hyper_encoder(topo_code))
            if ambient_code is not None:
                sdf_inputs.append(ambient_code)

            sdf_nn_output = self.sdf_network(torch.cat(sdf_inputs, dim=-1))
            sdf, feature_vector = sdf_nn_output[..., :1], sdf_nn_output[..., 1:]
            normals_can = torch.autograd.grad(sdf, points_can, grad_outputs=torch.ones_like(sdf, requires_grad=False, device=sdf.device), create_graph=True, retain_graph=True, only_inputs=True)[0]

        color_input = [points_can, rays_d_can, normals_can, feature_vector]
        if ambient_code is not None:
            color_input.append(ambient_code)
        
        raw_rgb = self.color_network(torch.cat(color_input, dim=-1))
        raw = torch.cat((raw_rgb, sdf), dim=-1)

        return raw, normals_can

    def get_alpha(self, sdf, normal, dirs, dists):
        # sdf.shape = [N, 1]
        # normal.shape = [N, 1]
        # dirs.shape = [N, 1]
        # dists.shape = [N, 1]
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6).squeeze()           # Single parameter

        true_cos = (dirs.reshape(-1, 3)*normal.reshape(-1, 3)).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(torch.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     torch.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf.reshape(-1, 1) + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf.reshape(-1, 1) - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha.reshape(sdf.shape)


flow_net_dict = {
    'mlp_flow': DNeRFWarp,
    'mlp_rt': SE3FieldNerfies,
}
hyper_net_dict = {
    'hypernerf': HyperNeRFAmbientNetwork,
}

@models.register('DynamicNeRF')
class DynamicNeRF(BaseModel):
    def setup(self):
     
        self.time_max = self.config.metadata.time_max
        self.N_samples = self.config.sampling.N_samples
        self.N_importance = self.config.sampling.N_importance

        self.register_buffer('scene_aabb', torch.as_tensor(self.config.metadata.scene_aabb, dtype=torch.float32))
        
        range = (self.scene_aabb[3:6] - self.scene_aabb[:3]).max().item()
        self.raw_noise_std = range/self.N_samples

        self.randomized = self.config.randomized
        self.background_color = None
        
        assert self.config.background == 'black'

        # create networks
        self.ambient_dim = self.config.get('ambient_dim', 0)
        self.deform_dim = self.config.get('deform_dim', 0) 
        self.hyper_dim = self.config.get('hyper_dim', 0)

        if self.ambient_dim > 0:
            self.register_parameter('ambient_codes', torch.nn.Parameter(torch.randn(self.time_max+1, self.ambient_dim)))
        else:
            self.ambient_codes = None

        if self.deform_dim > 0:
            self.register_parameter('deform_codes', torch.nn.Parameter(torch.randn(self.time_max+1, self.deform_dim)))
        else:
            self.deform_codes = None

        if self.config.get('flow_net', None):
            flow_net_name = self.config.flow_net.get('name', None)
            self.flow_net = None if flow_net_name is None else flow_net_dict[flow_net_name](**self.config.flow_net.args)
        else:
            self.flow_net = None

        if self.hyper_dim > 0 and self.config.get('hyper_net', None) is not None:
            hyper_net_name = self.config.hyper_net.get('name', None)
            hyper_net_name = self.config.hyper_net.get('name', None)
            self.hyper_net = None if hyper_net_name is None else hyper_net_dict[hyper_net_name](**self.config.hyper_net.args)
            self.hyper_dim = 0 if self.hyper_net is None else self.hyper_net.output_dim
        else:
            self.hyper_net = None

        # create NeRF-based MLPs
        coarse_type = self.config.coarse_model.get('type', 'AmbientNeRFMLP')
        nerf_models = dict(coarse=eval(coarse_type)(n_frames=self.time_max+1, **self.config.coarse_model.args))
        
        if self.N_importance > 0 and self.config.get('fine_model', False) is not None:
            fine_type = self.config.fine_model.get('type', 'AmbientNeRFMLP')
            nerf_models['fine'] =  eval(fine_type)(n_frames=self.time_max+1, **self.config.fine_model.args)

        self.nerf = torch.nn.ModuleDict(nerf_models)            

    def forward(self, rays, **kwargs):
        if self.training:
            out = self.forward_(rays, **kwargs)
        else:
            out = chunk_batch_levels(self.forward_, self.config.ray_chunk, rays, **kwargs)
        return {
            **out,
        }

    def _deform(self, pts_o, ray_d_o, deform_code):
        """
        Args:
            pts_o (N_rays, N_samples, 3): points in the observation space.
            ray_d_o (N_rays, N_samples, 3): Ray direction in the observation space
            deform_code (N_rays, N_samples, F)
            time_index ((1,) or (N_rays))
        """
        N_rays, N_samples, _ = pts_o.shape

        if deform_code is not None and self.flow_net is not None:
            pts_o = pts_o.reshape(-1, pts_o.shape[-1])
            ray_d_o = ray_d_o.reshape(-1, ray_d_o.shape[-1])
            with torch.inference_mode(False), torch.enable_grad():
                if not self.training:
                    pts_o = pts_o.clone() # points may be in inference mode, get a copy to enable grad
                pts_o.requires_grad_(True)
                deform_code = deform_code.reshape(-1, deform_code.shape[-1])

                pts_can = self.flow_net(pts_o, deform_code) # N,3
                # estimate jacobian
                y_0, y_1, y_2 = pts_can.split([1,1,1], dim=-1)
                _do = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
                grad_0 = torch.autograd.grad(y_0, inputs=pts_o, grad_outputs=_do, create_graph=True, retain_graph=True, only_inputs=True)[0].unsqueeze(1)
                grad_1 = torch.autograd.grad(y_1, inputs=pts_o, grad_outputs=_do, create_graph=True, retain_graph=True, only_inputs=True)[0].unsqueeze(1)
                grad_2 = torch.autograd.grad(y_2, inputs=pts_o, grad_outputs=_do, create_graph=True, retain_graph=True, only_inputs=True)[0].unsqueeze(1)
            pts_jacobian =  torch.cat([grad_0, grad_1, grad_2], dim=1) # (batch_size, dim_out, dim_in)
            ray_d_can = (pts_jacobian @ ray_d_o.unsqueeze(-1)).squeeze(-1) # view in observation space
            ray_d_can = torch.nn.functional.normalize(ray_d_can, dim=-1, p=2)

            pts_can = pts_o.reshape(N_rays, N_samples, 3)
            ray_d_can = ray_d_can.reshape(N_rays, N_samples, 3)
            pts_jacobian = pts_jacobian.reshape(N_rays, N_samples, 3, 3)
        else:
            pts_can = pts_o
            ray_d_can = ray_d_o
            pts_jacobian = None

        if self.hyper_net is not None:
            topo_code = self.hyper_net(pts_o, deform_code)
            topo_code = topo_code.reshape(N_rays, N_samples, topo_code.shape[-1])
        else:
            topo_code = None

        return pts_can, ray_d_can, topo_code, pts_jacobian

    def _nerf_forward(self, nerf_model, pts_o, ray_d_o, deform_code, ambient_code, z_vals, frame_id):
        if deform_code is not None:
            deform_code = deform_code[:, None].expand(-1, pts_o.shape[1], -1)

        if ambient_code is not None:
            ambient_code = ambient_code[:, None].expand(-1, pts_o.shape[1], -1)
        ray_d_o = ray_d_o[:, None].expand(-1, pts_o.shape[1], -1)

        pts_can, ray_d_can, topo_code, pts_jacobian = self._deform(pts_o, ray_d_o, deform_code)

        add_noise = lambda raw: raw + torch.randn_like(raw) * self.raw_noise_std if self.raw_noise_std > 0 else raw
        if isinstance(nerf_model, AmbientNeRFMLP):
            raw2alpha = lambda raw, dists, act_fn=torch.relu: 1.-torch.exp(-act_fn(add_noise(raw))*dists)
            raw = nerf_model(
                x=pts_can,
                topo_code=topo_code,
                ambient_code=ambient_code,
                rays_d=ray_d_can,
                frame_id=frame_id) # N_rays, N_samples, 4
            output = raw2outputs(raw, z_vals, raw2alpha, white_bkgd=False, use_sample_at_infinity=True)
        elif isinstance(nerf_model, AmbientNeuSMLP):
            raw, normals_can = nerf_model(
                points_can=pts_can,
                rays_d_can=ray_d_can,
                topo_code=topo_code,
                ambient_code=ambient_code,
            ) # N_rays, N_samples, 4
            normals_o = (torch.inverse(pts_jacobian) @ normals_can.unsqueeze(-1)).squeeze(-1)
            raw2alpha = lambda sdf, dists: nerf_model.get_alpha(add_noise(sdf), normals_o, ray_d_o, dists)
            output = raw2outputs(raw, z_vals, raw2alpha, normals=normals_o, white_bkgd=False, use_sample_at_infinity=False)
        else:
            raise NotImplementedError

        return raw, output

    def forward_(self, rays, **kwargs):
        to_ret = {}
        N_rays = rays.shape[0]
        near, far = kwargs.get('near', None), kwargs.get('far', None)
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        time_index = kwargs['time_index']
        frame_id = time_index

        t_len = time_index.numel() == 1
        ambient_code = None
        if self.ambient_codes is not None:
            ambient_code = self.ambient_codes[time_index]  # N_rays, F
            ambient_code = ambient_code.reshape(1, -1).expand(N_rays, -1) if t_len else ambient_code

        deform_code = None
        if self.deform_codes is not None:
            deform_code = self.deform_codes[time_index]
            deform_code = deform_code.reshape(1, -1).expand(N_rays, -1) if t_len else deform_code

        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=rays.device)
        z_vals = (near * (1.-t_vals) + far * (t_vals)).expand([N_rays, self.N_samples])

        if self.randomized: # stratified
            z_vals = resample_zdist(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        _, to_ret['coarse'] = self._nerf_forward(self.nerf['coarse'], pts, rays_d, deform_code, ambient_code, z_vals, frame_id=frame_id) # N_rays, N_samples, 4

        if self.N_importance > 0:
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, to_ret['coarse']['weights'][...,1:-1], self.N_importance, det=not self.randomized).detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            _, to_ret['fine'] = self._nerf_forward(self.nerf['fine'], pts, rays_d, deform_code, ambient_code, z_vals, frame_id=frame_id) # N_rays, N_samples, 4
            to_ret['fine']['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        return to_ret

    def isosurface(self, resolution=None):
        return
