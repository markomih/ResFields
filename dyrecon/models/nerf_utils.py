import torch

import math
import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn

from utils.misc import get_rank

from utils.misc import config_to_primitive
from models.utils import get_activation

import functools
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
        independent_layers: list = [],
        composition_rank: int = 10,
        n_frames: int = 1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init
        self.independent_layers = independent_layers
        self.composition_rank = composition_rank
        self.n_frames = n_frames

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()
        for ind_layer in self.independent_layers:
            lin = self.hidden_layers[i]
            self.register_parameter(f'lin_{ind_layer}_w1', torch.nn.Parameter(0.01*torch.randn((self.composition_rank, self.n_frames))))
            self.register_parameter(f'lin_{ind_layer}_m1', torch.nn.Parameter(0.01*torch.randn((self.composition_rank, lin.weight.shape[0], lin.weight.shape[1]))))

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x, frame_id=None):
        inputs = x
        for i in range(self.net_depth):
            lin = self.hidden_layers[i]
            if i in self.independent_layers:
                _w1 = getattr(self, f'lin_{i}_w1')  # R, T
                _m = getattr(self, f'lin_{i}_m1') #+ lin.weight[None] # R, F_out,F_in
                lin_w = (_w1[:, frame_id, None, None] * _m).sum(0) + lin.weight
                x = torch.nn.functional.linear(x, lin_w, lin.bias)
                # x = (lin_w @ x.permute(1,0)).permute(1, 0) + lin.bias[None] # F_out,F_in @ F_in,N -> F_out,N -> N,F_out
            else:
                x = lin(x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )

class HyperNeRFAmbientNetwork(nn.Module):
    """ Ambient Network inspired by HyperNeRF
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_feature: int = 64,  # ambient code length
        net_depth: int = 7, # The depth of the MLP.
        net_width: int = 64, # The width of the MLP.
        skip_layer: int = 4, # The layer to add skip layers to.
        output_dim: int = 2
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.posi_encoder = SinusoidalEncoder(input_dim, 0, 10, True)

        self.base = MLP(
            input_dim=self.posi_encoder.latent_dim + d_feature,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=True,
            output_dim=output_dim
        )
    def forward(self, x, feature):
        inputs = torch.cat((
            self.posi_encoder(x),
            feature,
        ), dim=-1)
        output = self.base(inputs)
        return output

class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        independent_layers: list = [],
        composition_rank: int = 10,
        n_frames: int = 1000,
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
            independent_layers=independent_layers,
            composition_rank=composition_rank,
            n_frames=n_frames,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None, frame_id=None):
        x = self.base(x, frame_id=frame_id)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma

class VarianceNetwork(nn.Module):
    def __init__(self, init_val:float = 0.3):
        super(VarianceNetwork, self).__init__()
        self.init_val = init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class DNeRFWarp(nn.Module):
    def __init__(self, d_feature) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)

        self.warp = MLP(
            input_dim=self.posi_encoder.latent_dim+d_feature,
            output_dim=3,
            net_depth=4,
            net_width=64,
            skip_layer=2,
            output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        )

    def forward(self, x, t):
        input = torch.cat((self.posi_encoder(x), t), dim=-1)
        x = x + self.warp(input)
        return x


class SE3FieldNerfies(torch.nn.Module):
    """ Model deformation like in Nerfies/HyperNeRF. Network that predicts warps as an SE(3) field.
    """

    def __init__(self,
            d_feature = 64,
            activation = torch.relu,
            skip_layer: int = 4,
            trunk_depth: int = 6,
            trunk_width: int = 128,
            rotation_depth: int = 0,
            rotation_width: int = 128,
            pivot_depth: int = 0,
            pivot_width: int = 128,
        ):
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)

        self.trunk = MLP(
            input_dim=self.posi_encoder.latent_dim + d_feature,
            net_depth=trunk_depth,
            net_width=trunk_width,
            hidden_activation=activation,
            output_enabled=False,
            skip_layer=skip_layer)

        self.branches_w = MLP(
            input_dim=trunk_width,
            net_depth=rotation_depth,
            net_width=rotation_width,
            output_enabled=True, output_dim=3,
            hidden_activation=activation,
        )
        self.branches_v = MLP(
            input_dim=trunk_width,
            net_depth=pivot_depth,
            net_width=pivot_width,
            output_enabled=True, output_dim=3,
            hidden_activation=activation,
        )

    def forward(self, points, deform_code):
        # points.shape = (-1, 3)
        # deform_code.shape = (-1, F)
        inputs = torch.cat([
            self.posi_encoder(points),
            deform_code
        ], axis=-1)
        trunk_output = self.trunk(inputs)

        w = self.branches_w(trunk_output)
        v = self.branches_v(trunk_output)
        theta = torch.norm(w, dim=-1, p=2, keepdim=True)
        w = w / (theta+1e-6)
        v = v / (theta+1e-6)
        screw_axis = torch.cat([w, v], axis=-1)
        transform_R, transform_t = self.exp3(screw_axis, theta)

        warped_points = (transform_R @ points.unsqueeze(-1)).squeeze(-1) + transform_t

        return warped_points

    @staticmethod
    def exp3(x, theta): # x.shape = [B,6]; theta.shape=[B,1]
        # x_ = x.view(-1, 6)
        w, v = x.split([3, 3], dim=1)
        W = SE3FieldNerfies.skew_mat(w)  # B,3,3
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
        return R, p.reshape(-1, 3)

    @staticmethod
    def skew_mat(x: torch.Tensor):
        # size: [*, 3] -> [*, 3, 3]
        x_ = x.view(-1, 3)
        x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
        O = torch.zeros_like(x1)

        X = torch.stack((torch.stack(
            (O, -x3, x2), dim=1), torch.stack(
                (x3, O, -x1), dim=1), torch.stack((-x2, x1, O), dim=1)),
                        dim=1)
        return X.view(*(x.size()[0:-1]), 3, 3)

class DNeRFRadianceField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp = MLP(
            input_dim=self.posi_encoder.latent_dim
            + self.time_encoder.latent_dim,
            output_dim=3,
            net_depth=4,
            net_width=64,
            skip_layer=2,
            output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        )
        self.nerf = VanillaNeRFRadianceField()

    def query_can_density(self, x):
        return self.nerf.query_density(x)

    def query_density(self, x, t):
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
        )
        return self.nerf.query_density(x)

    def forward(self, x, t, condition=None):
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
        )
        return self.nerf(x, condition=condition)

class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.update_step(None, None) # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq*x) * mask]                
        return torch.cat(out, -1)          

    def update_step(self, epoch, global_step):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
            rank_zero_debug(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
    
    def forward(self, x, *args):
        return self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)


def get_encoding(n_input_dims, config):
    # input suppose to be range [0, 1]
    if config.otype == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config), dtype=torch.float32)
    encoding = CompositeEncoding(encoding, include_xyz=config.get('include_xyz', False), xyz_scale=2., xyz_offset=-1.)
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])
    
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
    rank_zero_debug('Initialize tcnn MLP to approximately represent a sphere.')
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    padto = 16 if config.otype == 'FullyFusedMLP' else 8
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons), std=0.0001)
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)


def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
            if config.get('sphere_init', False):
                sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    return network


def get_encoding_with_network(n_input_dims, n_output_dims, encoding_config, network_config):
    # input suppose to be range [0, 1]
    if encoding_config.otype in ['VanillaFrequency'] \
        or network_config.otype in ['VanillaMLP']:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        encoding_with_network = nn.Sequential(
            encoding,
            network
        )
    else:
        with torch.cuda.device(get_rank()):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config)
            )
    return encoding_with_network

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


def raw2outputs(raw, z_vals, raw2alpha, normals=None, white_bkgd=False, use_sample_at_infinity=True, rgb_act=torch.sigmoid):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    dists = z_vals[...,1:] - z_vals[...,:-1]
    # TODO: check if sample at infity is needed
    last_sample_t = 1e10 if use_sample_at_infinity else 1e-19
    dists = torch.cat([dists, torch.tensor(last_sample_t, device=raw.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    rgb = rgb_act(raw[...,:3])  # [N_rays, N_samples, 3]
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    normal_map = None
    if normals is not None:
        normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
        normal_map = torch.sum(weights[...,None]*normals, -2)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    to_ret = {
        'rgb': rgb_map,
        'opacity': acc_map,
        'depth': depth_map,
        'disp': disp_map,
        'tvals': z_vals,
        'weights': weights,
    }
    if normal_map is not None:
        to_ret['normal'] = normal_map
    return to_ret

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., device=bins.device, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def resample_zdist(z_vals):
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    
    t_rand = torch.rand(z_vals.shape, device=z_vals.device)
    z_vals = lower + (upper - lower) * t_rand
    return z_vals