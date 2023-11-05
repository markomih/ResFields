import torch
import models
import torch.nn.functional as F

from models.base import BaseModel

@models.register('DynamicFields')
class DynamicFields(BaseModel):
    def setup(self):
        self.n_frames = self.config.n_frames
        
        # create networks
        self.ambient_dim = self.config.get('ambient_dim', 0)
        self.deform_dim = self.config.get('deform_dim', 0) 
        
        self.ambient_codes = torch.nn.Parameter(torch.randn(self.n_frames, self.ambient_dim)) if self.ambient_dim > 0 else None
        self.deform_codes = torch.nn.Parameter(torch.randn(self.n_frames, self.deform_dim)) if self.deform_dim > 0 else None
        self.deform_net = models.make(self.config.deform_net.name, self.config.deform_net) if self.config.deform_net else None
        self.hyper_net = models.make(self.config.hyper_net.name, self.config.hyper_net) if self.config.hyper_net else None

        self.config.sdf_net.d_in_2 = self.hyper_net.out_dim
        self.config.sdf_net.n_frames = self.n_frames
        self.sdf_net = models.make(self.config.sdf_net.name, self.config.sdf_net)
        self.color_net = models.make(self.config.color_net.name, self.config.color_net)
    
    def forward(self, pts, pts_time, frame_id, rays_d, alpha_ratio=1.0, estimate_normals=True, estimate_color=True):
        """ Query the model at the specified points in space and time.

        Args:
            pts: (n_rays, n_samples, 3)
            pts_time: (n_rays, n_samples, 1)
            frame_id: (n_rays)
            rays_d: (n_rays, n_samples, 3)
            alpha_ratio (float): alpha ratio for traning
            estimate_normals (bool): whether to estimate normals
            estimate_color (bool): whether to estimate color
        Returns:
            to_ret (dict): dictionary of outputs
                sdf: (n_rays, n_samples, 1)
                color: (n_rays, n_samples, 3)
                gradients_o: (n_rays, n_samples, 3)
                normal: (n_rays, n_samples, 3)
        """
        grad_enabled = self.training or estimate_normals
        with torch.inference_mode(not grad_enabled), torch.set_grad_enabled(grad_enabled):  # enable gradient for computing gradients
            if estimate_normals:
                if not self.training:
                    pts = pts.clone()
                pts.requires_grad_(True)

            deform_codes = self.deform_codes[frame_id] if self.deform_codes is not None else None
            if deform_codes is not None:
                deform_codes = deform_codes.view(-1, 1, deform_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)
            ambient_codes = self.ambient_codes[frame_id] if self.ambient_codes is not None else None
            if ambient_codes is not None:
                ambient_codes = ambient_codes.view(-1, 1, ambient_codes.shape[-1]).expand(pts.shape[0], pts.shape[1], -1)

            pts_canonical = pts if self.deform_net is None else self.deform_net(deform_codes, pts, alpha_ratio, pts_time)
            hyper_coord = self.hyper_net(deform_codes, pts, pts_time, alpha_ratio)

            sdf_nn_output = self.sdf_net(pts_canonical, hyper_coord, alpha_ratio, input_time=pts_time.squeeze(-1), frame_id=frame_id.squeeze(-1))
            sdf, feature_vector = sdf_nn_output[..., :1], sdf_nn_output[..., 1:] # (n_rays, n_samples, 1), (n_rays, n_samples, F)
            to_ret = {'sdf': sdf}

            if estimate_normals:
                to_ret['gradients_o'] =  torch.autograd.grad(outputs=sdf, inputs=pts, grad_outputs=torch.ones_like(sdf, requires_grad=False, device=sdf.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
                to_ret['normal'] = F.normalize(to_ret['gradients_o'], dim=-1, p=2)

        if estimate_color:
            to_ret['color'] = self.color_net(feature=feature_vector, point=pts_canonical, ambient_code=ambient_codes, view_dir=rays_d, normal=to_ret.get('gradients_o', None), alpha_ratio=alpha_ratio) # n_rays, n_samples, 3

        return to_ret
