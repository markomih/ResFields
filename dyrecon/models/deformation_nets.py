import torch
import models
from models.base import BaseModel
from .fields import get_embedder
import torch.nn as nn
import numpy as np

class BaseDeformNetwork(BaseModel):
    def forward(self, deformation_code, input_pts, alpha_ratio, time_step=None):
        raise NotImplementedError

# Deform
@models.register('deformation_NDR')
class DeformNetwork(BaseDeformNetwork):
    def setup(self):
        d_feature = self.config.d_feature
        d_in = self.config.d_in
        d_out_1 = self.config.d_out_1
        d_out_2 = self.config.d_out_2
        n_blocks = self.config.n_blocks
        d_hidden = self.config.d_hidden
        n_layers = self.config.n_layers
        skip_in = self.config.skip_in
        multires = self.config.multires
        weight_norm = self.config.weight_norm

        self.n_blocks = n_blocks
        self.skip_in = skip_in

        # part a
        # xy -> z
        ori_in = d_in - 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_a = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_a - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_a - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_a - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_a_"+str(l), lin)

        # part b
        # z -> xy
        ori_in = 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden] + [d_out_2]

        self.embed_fn_2 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_2 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_b = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_b - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_b - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_b - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_b_"+str(l), lin)

        # latent code
        for i_b in range(self.n_blocks):
            lin = nn.Linear(d_feature, d_feature)
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.constant_(lin.weight, 0.0)
            setattr(self, "lin"+str(i_b)+"_c", lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, deformation_code, input_pts, alpha_ratio, time_step=None):
        x = input_pts
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            # part a
            if form == 0:
                # zyx
                if mode == 0:
                    x_focus = x[..., [2]]
                    x_other = x[..., [0,1]]
                elif mode == 1:
                    x_focus = x[..., [1]]
                    x_other = x[..., [0,2]]
                else:
                    x_focus = x[..., [0]]
                    x_other = x[..., [1,2]]
            else:
                # xyz
                if mode == 0:
                    x_focus = x[..., [0]]
                    x_other = x[..., [1,2]]
                elif mode == 1:
                    x_focus = x[..., [1]]
                    x_other = x[..., [0,2]]
                else:
                    x_focus = x[..., [2]]
                    x_other = x[..., [0,1]]
            x_ori = x_other # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_other = self.embed_fn_1(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_focus = x_focus - x

            # part b
            x_focus_ori = x_focus # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_focus = self.embed_fn_2(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], -1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = self.euler2rot_2dinv(x[..., [0]])
            trans_2d = x[..., 1:]
            x_other = (rot_2d @ (x_ori - trans_2d)[..., None]).squeeze(-1)
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[...,[0]], x_focus_ori, x_other[...,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[...,[0]], x_focus_ori, x_other[...,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)

        return x

    @staticmethod
    def euler2rot_2dinv(euler_angle):
        # (B, S, 1) -> (B, S, 2, 2)
        theta = euler_angle.unsqueeze(-1) # B, S, 1, 1
        rot = torch.cat((
            torch.cat((theta.cos(), -theta.sin()), -2),
            torch.cat((theta.sin(), theta.cos()), -2),
        ), -1)

        return rot

    def inverse(self, deformation_code, input_pts, alpha_ratio, time_step=None):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            i_b = self.n_blocks - 1 - i_b # inverse
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part b
            if form == 0:
                # axis: z -> y -> x
                if mode == 0:
                    x_focus = x[..., [0,1]]
                    x_other = x[..., [2]]
                elif mode == 1:
                    x_focus = x[..., [0,2]]
                    x_other = x[..., [1]]
                else:
                    x_focus = x[..., [1,2]]
                    x_other = x[..., [0]]
            else:
                # axis: x -> y -> z
                if mode == 0:
                    x_focus = x[..., [1,2]]
                    x_other = x[..., [0]]
                elif mode == 1:
                    x_focus = x[..., [0,2]]
                    x_other = x[..., [1]]
                else:
                    x_focus = x[..., [0,1]]
                    x_other = x[..., [2]]
            x_ori = x_other # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_other = self.embed_fn_2(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2d(x[..., [0]])
            trans_2d = x[..., 1:]
            x_focus = torch.bmm(rot_2d, x_focus[...,None]).squeeze(-1) + trans_2d

            # part a
            x_focus_ori = x_focus # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_focus = self.embed_fn_1(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_other = x_ori + x
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[...,[0]], x_other, x_focus_ori[...,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[...,[0]], x_other, x_focus_ori[...,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)

        return x


# Models
@models.register('deformation_DNeRF')
class DeformDNeRF(BaseDeformNetwork):
    """ Model deformation like in DNeRF 
    """

    def setup(self):
        d_time = self.config.d_time
        d_in = self.config.d_in
        depth = self.config.depth
        width = self.config.width
        skips = self.config.skips
        multires = self.config.multires
        if multires > 0:
            self.embed_fn, d_in = get_embedder(multires, input_dims=d_in)
        else:
            self.embed_fn = lambda x, y: x

        self.time_mlp = MLP(
            in_ch=d_in+d_time,
            depth=depth,
            width=width,
            skips=skips,
            out_ch=d_in,
            act=torch.relu
        )

    def forward(self, deformation_code, input_pts, alpha_ratio, time_step):
        points = self.embed_fn(input_pts, alpha_ratio)
        input_x = torch.cat((points, time_step), dim=-1)
        dx = self.time_mlp(input_x)
        return input_pts + dx


@models.register('deformation_SE3Field')
class SE3Field(BaseDeformNetwork):
    """ Model deformation like in Nerfies/HyperNeRF. Network that predicts warps as an SE(3) field.

    Attributes:
        points_encoder: the positional encoder for the points.
        metadata_encoder: an encoder for metadata.
        alpha: the alpha for the positional encoding.
        skips: the index of the layers with skip connections.
        depth: the depth of the network excluding the logit layer.
        hidden_channels: the width of the network hidden layers.
        activation: the activation for each layer.
        metadata_encoded: whether the metadata parameter is pre-encoded or not.
        hidden_initializer: the initializer for the hidden layers.
        output_initializer: the initializer for the last logit layer.
    """
    def setup(self):
        d_feature = self.config.d_feature
        d_in = self.config.d_in
        multires = self.config.multires

        activation = self.config.get('activation', torch.relu)
        skips: tuple = self.config.get('skips', (4,))
        trunk_depth: int = self.config.get('trunk_depth', 6)
        trunk_width: int = self.config.get('trunk_width', 128)
        rotation_depth: int = self.config.get('rotation_depth', 0)
        rotation_width: int = self.config.get('rotation_width', 128)
        pivot_depth: int = self.config.get('pivot_depth', 0)
        pivot_width: int = self.config.get('pivot_width', 128)

        if multires > 0:
            self.embed_fn, d_in = get_embedder(multires, input_dims=d_in)
        else:
            self.embed_fn = lambda x, y: x

        self.trunk = MLP(
            in_ch=d_in + d_feature,
            depth=trunk_depth,
            width=trunk_width,
            act=activation,
            skips=skips)

        self.branches_w = MLP(
            in_ch=trunk_width,
            depth=rotation_depth,
            width=rotation_width,
            out_ch=3,
            act=activation,
        )
        self.branches_v = MLP(
            in_ch=trunk_width,
            depth=pivot_depth,
            width=pivot_width,
            out_ch=3,
            act=activation,
        )

    def forward(self, deform_code, points, alpha_ratio, time_step=None):
        inputs = torch.cat([
            self.embed_fn(points, alpha_ratio),
            deform_code
        ], axis=-1)
        trunk_output = self.trunk(inputs)

        w = self.branches_w(trunk_output)
        v = self.branches_v(trunk_output)
        theta = torch.norm(w, dim=-1, p=2, keepdim=True)
        w = w / (theta+1e-6)
        v = v / (theta+1e-6)
        screw_axis = torch.cat([w, v], axis=-1)
        transform_R, transform_t = self.exp3(screw_axis, theta) # B,S,3,3; B,S,3

        warped_points = (transform_R @ points.unsqueeze(-1)).squeeze(-1) + transform_t

        return warped_points

    @staticmethod
    def exp3(x, theta): # x.shape = [B,6]; theta.shape=[B,1]
        B, _S, dim = x.shape
        B, _S, _ = theta.shape
        x, theta = x.view(-1, 6), theta.view(-1, 1)

        # x_ = x.view(-1, 6)
        w, v = x.split([3, 3], dim=1)
        W = SE3Field.skew_mat(w)  # B,3,3
        S = W.bmm(W)  # B,3,3
        I = torch.eye(3).to(w).unsqueeze(0) # 1,3,3
        cos_t = torch.cos(theta).unsqueeze(-1) # B,1,1
        sin_t = torch.sin(theta).unsqueeze(-1) # B,1,1
        theta = theta.unsqueeze(-1) # B,1,1

        # Rodrigues' rotation formula.
        # R = I + sin(t)*W + (1-cos(t))*(w*w')

        R = I + sin_t * W + (1.0 - cos_t) * S
        # p = (I*t + (1-cos(t))*w + (t- sin(t))*(w**2))*v

        _p = I*theta + (1.0 - cos_t)*W + (theta - sin_t)*S
        p = torch.bmm(_p, v.reshape(-1, 3, 1))
        p = p.reshape(-1, 3)
        return R.view(B, _S, *R.shape[1:]), p.view(B, _S, *p.shape[1:])

    @staticmethod
    def skew_mat(x: torch.tensor):
        # size: [*, 3] -> [*, 3, 3]
        x_ = x.view(-1, 3)
        x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
        O = torch.zeros_like(x1)

        X = torch.stack((torch.stack(
            (O, -x3, x2), dim=1), torch.stack(
                (x3, O, -x1), dim=1), torch.stack((-x2, x1, O), dim=1)),
                        dim=1)
        return X.view(*(x.size()[0:-1]), 3, 3)

class MLP(torch.nn.Module):
    def __init__(self,
            in_ch: int,
            depth: int, 
            width: int, 
            out_ch: int = 0,
            skips: tuple = tuple(), 
            act=torch.relu, 
            out_act=lambda x: x
        ) -> None:
        super().__init__()
        self.act = act
        self.out_act = out_act
        self.skips = skips

        layers = [torch.nn.Linear(in_ch, width)]
        for i in range(depth - 1):
            in_channels = width
            if i in skips:
                in_channels += in_ch

            layers.append(torch.nn.Linear(in_channels, width))
        self.net = torch.nn.ModuleList(layers)
        if out_ch > 0:
            self.net_out = torch.nn.Linear(width, out_ch)
        else:
            self.net_out = lambda x: x

    def forward(self, input_x):
        h = input_x
        for i, l in enumerate(self.net):
            h = self.net[i](h)
            h = self.act(h)
            if i in self.skips:
                h = torch.cat([input_x, h], -1)

        return self.out_act(self.net_out(h))

#