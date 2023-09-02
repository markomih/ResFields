import torch
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
import models
from models.base import BaseModel
import resfields

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) * math.pi
        else:
            freq_bands = torch.linspace(2.**0.*math.pi, 2.**max_freq*math.pi, self.num_freqs)

        self.num_fns = len(self.kwargs['periodic_fns'])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs, alpha_ratio=0.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        start = 0
        if self.include_input:
            start = 1
        for i in range(self.num_freqs):
            _dec = (1.-math.cos(math.pi*(max(min(alpha_ratio*self.num_freqs-i, 1.), 0.)))) * .5
            output[..., (self.num_fns*i+start)*self.input_dims:(self.num_fns*(i+1)+start)*self.input_dims] *= _dec
        return output


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, alpha_ratio, eo=embedder_obj): return eo.embed(x, alpha_ratio)
    return embed, embedder_obj.out_dim
    
@models.register('sdf_network')
class SDFNetwork(BaseModel):
    def setup(self):
        self.n_frames = self.config.n_frames
        self.capacity = self.n_frames
        self.d_out = self.config.d_out
        self.d_in_1 = self.config.d_in_1
        self.d_in_2 = self.config.d_in_2
        self.d_hidden = self.config.d_hidden
        self.n_layers = self.config.n_layers
        self.skip_in = self.config.skip_in
        self.multires = self.config.multires
        self.multires_topo = self.config.multires_topo
        self.bias = self.config.bias
        self.scale = self.config.scale
        self.geometric_init = self.config.geometric_init
        self.weight_norm = self.config.weight_norm
        self.inside_outside = self.config.inside_outside

        self.resfield_layers = self.config.get('resfield_layers', [])
        self.composition_rank = self.config.get('composition_rank', 10)
        # create nets
        dims = [self.d_in_1 + self.d_in_2] + [self.d_hidden for _ in range(self.n_layers)] + [self.d_out]

        self.embed_fn_fine = None
        self.embed_amb_fn = None

        input_ch_1 = self.d_in_1
        input_ch_2 = self.d_in_2
        if self.multires > 0:
            embed_fn, input_ch_1 = get_embedder(self.multires, input_dims=self.d_in_1)
            self.embed_fn_fine = embed_fn
            dims[0] += (input_ch_1 - self.d_in_1)
        if self.multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(self.multires_topo, input_dims=self.d_in_2)
            self.embed_amb_fn = embed_amb_fn
            dims[0] += (input_ch_2 - self.d_in_2)

        self.num_layers = len(dims)
        self.skip_in = self.skip_in
        self.scale = self.scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            _rank = self.composition_rank if l in self.resfield_layers else 0
            _capacity = self.capacity if l in self.resfield_layers else 0
            lin = resfields.Linear(dims[l], out_dim, rank=_rank, capacity=_capacity, mode='lookup')

            if self.geometric_init:
                if l == self.num_layers - 2:
                    if not self.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.bias)
                elif self.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, self.d_in_1:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :self.d_in_1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if self.multires > 0:
                        torch.nn.init.constant_(lin.weight[:, -(dims[0] - self.d_in_1):-input_ch_2], 0.0)
                    if self.multires_topo > 0:
                        torch.nn.init.constant_(lin.weight[:, -(input_ch_2 - self.d_in_2):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, input_pts, topo_coord=None, alpha_ratio=1.0, input_time=None, frame_id=None):
        """
            Args:
                input_pts: Tensor of shape (n_rays, n_samples, d_in_1)
                topo_coord: Optional tensor of shape (n_rays, n_samples, d_in_2)
                alpha_ratio (float): decay ratio of positional encoding
                input_time: Optional tensor of shape (n_rays, n_rays)
                # frame_id: Optional tensor of shape (n_rays)
        """
        # TIME = topo_coord
        inputs = input_pts * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs, alpha_ratio)
        if self.embed_amb_fn is not None:
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        if topo_coord is not None:
            inputs = torch.cat([inputs, topo_coord], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            lin = getattr(self, "lin" + str(l))
            x = lin(x, input_time=input_time, frame_id=frame_id)

            if l < self.num_layers - 2:
                x = self.activation(x)
        sdf = (x[..., :1] / self.scale)
        out = torch.cat([sdf, x[..., 1:]], dim=-1)
        return out


    # Anneal
    def sdf(self, x, topo_coord, alpha_ratio, **kwargs):
        return self.forward(x, topo_coord, alpha_ratio, **kwargs)[..., :1]

    def sdf_hidden_appearance(self, x, topo_coord, alpha_ratio, **kwargs):
        return self.forward(x, topo_coord, alpha_ratio, **kwargs)

    def gradient(self, x, topo_coord, alpha_ratio, **kwargs):
        x.requires_grad_(True)
        y = self.sdf(x, topo_coord, alpha_ratio, **kwargs)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

@models.register('color_network')
class ColorNetwork(BaseModel):
    def setup(self):
        supported_modes = ['feature', 'point', 'ambient', 'view', 'normal']

        # get params
        self.mode = self.config.mode # dictionary: encoding_type,feature_dim
        assert len(self.mode) != 0, 'No input features specified'
        d_hidden = self.config.d_hidden
        n_layers = self.config.n_layers

        weight_norm = self.config.get('weight_norm', False)
        d_out = self.config.get('d_out', 3)
        multires_view = self.config.get('multires_view', 0)

        # create encodings
        f_in = 0
        for encoding, encoding_dim in self.mode.items():
            assert encoding in supported_modes, f'Encoding {encoding} not supported'
            f_in += encoding_dim

        self.embedview_fn = None
        if multires_view > 0 and 'view' in self.mode:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            f_in += (input_ch - self.config.dim_view)

        # create network
        dims = [f_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, feature=None, point=None, ambient_code=None, view_dir=None, normal=None, alpha_ratio=1.0):
        # concatentate all inputs
        input_encoding = []
        if 'feature' in self.mode:
            input_encoding.append(feature)
        if 'point' in self.mode:
            input_encoding.append(point)
        if 'ambient' in self.mode:
            input_encoding.append(ambient_code)
        if 'view' in self.mode:
            if self.embedview_fn is not None:
                view_dirs = self.embedview_fn(view_dirs, alpha_ratio) # Anneal
            input_encoding.append(view_dir)
        if 'normal' in self.mode:
            input_encoding.append(normal)
        x = torch.cat(input_encoding, dim=-1)

        # forward through network
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = torch.sigmoid(x)
        return x


@models.register('laplace_density')
class LaplaceDensity(BaseModel):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def setup(self):
        beta = self.config.get('beta', 0.1)
        beta_min = self.config.get('beta_min', 0.0001)
        self.register_parameter('beta', torch.nn.Parameter(torch.tensor(beta)))
        self.register_buffer('beta_min', torch.tensor(beta_min))
        
    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

    def inv_s(self):
        return torch.reciprocal(self.get_beta())

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

class HyperNetwork(BaseModel):
    def setup(self):
        self.d_in = self.config.d_in
        self.multires_out = self.config.multires_out
        self.out_dim = self.config.d_out
        if self.multires_out > 0:
            self.embed_fn, self.out_dim = get_embedder(self.multires_out, input_dims=self.config.d_out)
        else:
            self.embed_fn = None

    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        raise NotImplementedError

    def forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        if self.d_in == 0:
            return None
        out = self._forward(deformation_code, input_pts, input_time, alpha_ratio)
        if self.embed_fn is not None:
            out = self.embed_fn(out, alpha_ratio)
        return out

@models.register('hyper_time_network')
class HyperTimeNetwork(HyperNetwork):
    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        return input_time

@models.register('hyper_cond_network')
class HyperCondNetwork(HyperNetwork):

    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        return deformation_code

@models.register('hyper_topo_network')
class TopoNetwork(HyperNetwork):
    # Adapted from NDR
    def setup(self):
        super().setup()
        d_feature = self.config.d_feature
        d_in = self.config.d_in
        d_out = self.config.d_out
        d_hidden = self.config.d_hidden
        n_layers = self.config.n_layers
        skip_in = self.config.skip_in 
        multires = self.config.multires
        bias = self.config.bias 
        weight_norm = self.config.weight_norm
        
        dims_in = d_in
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0.0, std=1e-5)
                torch.nn.init.constant_(lin.bias, bias)
            elif multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, d_in:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :d_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif multires > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims_in - d_in):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.activation = nn.Softplus(beta=100)


    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
        x = torch.cat([input_pts, deformation_code], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_pts], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

@models.register('siren_mlp')
class SirenMLP(BaseModel):
    def setup(self):
        in_features = self.config.in_features
        out_features = self.config.out_features
        hidden_features = self.config.hidden_features
        num_hidden_layers = self.config.num_hidden_layers
        # resfield parameters
        composition_rank = self.config.composition_rank
        resfield_layers = self.config.resfield_layers
        capacity = self.config.capacity
        mode = self.config.get('mode', 'lookup')
        coeff_ratio = self.config.get('coeff_ratio', 1.0)
        fuse_mode = self.config.get('fuse_mode', 'add')
        compression = self.config.compression

        dims = [in_features] + [hidden_features for _ in range(num_hidden_layers)] + [out_features]
        self.nl = Sine()
        self.net = []
        for i in range(len(dims) - 1):
            _rank = composition_rank if i in resfield_layers else 0
            _capacity = capacity if i in resfield_layers else 0
            if not isinstance(_rank, int) and compression != 'tucker':
                _rank = _rank[i]
            lin = resfields.Linear(dims[i], dims[i + 1], rank=_rank, capacity=_capacity, mode=mode, compression=compression, fuse_mode=fuse_mode, coeff_ratio=coeff_ratio)
            lin.apply(self.first_layer_sine_init if i == 0 else self.sine_init)
            self.net.append(lin)
        self.net = torch.nn.ModuleList(self.net)

    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
    
    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, coords, frame_id=None, input_time=None):
        x = coords
        for lin in self.net[:-1]:
            x = self.nl(lin(x, frame_id=frame_id, input_time=input_time))
            if lin.compression == 'resnet' and lin.capacity > 0:
                if frame_id.numel() == 1:
                    x = x + lin.resnet_vec[frame_id].view(1, 1, lin.resnet_vec.shape[-1])
                else:
                    x = x + lin.resnet_vec[:, None] # T, S, F_out
        x = self.net[-1](x, frame_id=frame_id, input_time=input_time)
        return x


@models.register('relu_mlp')
class ReluMLP(BaseModel):
    def setup(self):
        in_features = self.config.in_features
        out_features = self.config.out_features
        hidden_features = self.config.hidden_features
        num_hidden_layers = self.config.num_hidden_layers
        # resfield parameters
        composition_rank = self.config.composition_rank
        resfield_layers = self.config.resfield_layers
        capacity = self.config.capacity
        mode = self.config.get('mode', 'lookup')
        coeff_ratio = self.config.get('coeff_ratio', 1.0)
        fuse_mode = self.config.get('fuse_mode', 'add')
        compression = self.config.compression

        dims = [in_features] + [hidden_features for _ in range(num_hidden_layers)] + [out_features]
        self.nl = torch.nn.ReLU()
        self.net = []
        for i in range(len(dims) - 1):
            _rank = composition_rank if i in resfield_layers else 0
            _capacity = capacity if i in resfield_layers else 0

            lin = resfields.Linear(dims[i], dims[i + 1], rank=_rank, capacity=_capacity, mode=mode, compression=compression, fuse_mode=fuse_mode, coeff_ratio=coeff_ratio)
            lin.apply(self.init_weights_normal)
            self.net.append(lin)
        self.net = torch.nn.ModuleList(self.net)

    @staticmethod
    @torch.no_grad()
    def init_weights_normal(m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
    
    def forward(self, coords, frame_id=None, input_time=None):
        x = coords
        for lin in self.net[:-1]:
            x = self.nl(lin(x, frame_id=frame_id, input_time=input_time))
            if lin.compression == 'resnet' and lin.capacity > 0:
                if frame_id.numel() == 1:
                    x = x + lin.resnet_vec[frame_id].view(1, 1, lin.resnet_vec.shape[-1])
                else:
                    x = x + lin.resnet_vec[:, None] # T, S, F_out
        x = self.net[-1](x, frame_id=frame_id, input_time=input_time)
        return x

        
class Sine(nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

@models.register('ngp_mlp')
class NGPMLP(BaseModel):
    def setup(self):
        in_features = self.config.in_features
        out_features = self.config.out_features
        hidden_features = self.config.get('hidden_features', 64)
        log2_hashmap_size = self.config.get('log2_hashmap_size', 20)
        num_hidden_layers = self.config.get('num_hidden_layers', 2)
        n_levels = self.config.get('n_levels', 16)

        config = {
            'encoding': {
                'otype': 'HashGrid', 
                'n_levels': n_levels, 
                'n_features_per_level': 2, 
                'log2_hashmap_size': log2_hashmap_size, # 2**14 - 2**24
                'base_resolution': 16, 
                'per_level_scale': 1.5
                }, 
            'network': {
                'otype': 'FullyFusedMLP', 
                'activation': 'ReLU', 
                'output_activation': 'None', 
                'n_neurons': hidden_features, 
                'n_hidden_layers': num_hidden_layers
            }
        }
        import tinycudann as tcnn
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=in_features,
            n_output_dims=out_features,
            encoding_config=config['encoding'],
            network_config=config['network']
        )

    def forward(self, coords, frame_id=None, input_time=None):
        # coords: (n_points, dim) # range (-1, 1)
        coords = coords * 0.5 + 0.5  # rescale to (0, 1)
        if len(coords.shape) == 3:
            B, T, d = coords.shape
        output = self.net(coords.view(-1, coords.shape[-1]))
        if len(coords.shape) == 3:
            output = output.view(B, T, output.shape[-1])
        return output
