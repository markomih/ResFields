import gc
from collections import defaultdict

import torch
import numpy as np
import mcubes
import trimesh
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from packaging.version import parse as parse_version

def chunk_batch(func, chunk_size, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            out[k].append(v if torch.is_grad_enabled() else v.detach())
    
    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out

def chunk_batch_levels(func, chunk_size, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = {}
    out_type = None
    for i in range(0, B, chunk_size):
        _out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        levels = list(_out_chunk.keys())
        for level in levels:
            if level not in out:
                out[level] = defaultdict(list)
            out_chunk = _out_chunk[level]
            if out_chunk is None:
                continue
            out_type = type(out_chunk)
            if isinstance(out_chunk, torch.Tensor):
                out_chunk = {0: out_chunk}
            elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
                chunk_length = len(out_chunk)
                out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
            elif isinstance(out_chunk, dict):
                pass
            else:
                print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
                exit(1)
            for k, v in out_chunk.items():
                out[level][k].append(v.clone().detach() if torch.is_tensor(v) else v)
        del _out_chunk
    
    if out_type is None:
        return

    for level in levels:
        out[level] = {k: torch.cat(v, dim=0) for k, v in out[level].items()}
    return out

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    name = name.lower()
    if name is None or name == 'none':
        return nn.Identity()
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name.lower() == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name.lower() == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x

def masked_mean(x:torch.Tensor, mask: torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """Compute mean of masked values by soft blending.

    Args:
        x (types.Array): Input array of shape (...,).
        mask (types.Array): Mask array in [0, 1]. Shape will be broadcasted to
            match x.

    Returns:
        types.Array: Masked mean of x of shape ().
    """
    if mask is None:
        return x.mean()

    mask = torch.broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(eps)  # type: ignore

def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if parse_version(torch.__version__) < parse_version('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def mape_loss(pred, target, reduction='mean'):
    # pred, target: [B, 1], torch tensor
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == 'mean':
        loss = loss.mean()
    
    return loss

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    if len(vertices) > 0:
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    device = bound_max.device
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def ray_bbox_intersection(bounds, orig, direct, boffset=(-0.01, 0.01)):
    ''' Calculate the intersections of rays and the 3d bounding box. 
    Adapted from KeypointNeRF (ECCV 2022)
    Args:
        bounds: (2, 3): min, max
        orig: (N, 3): origin of rays
        direct: (N, 3): direction of rays
    return:
        near (N - points outside the box): the start of the ray inside the box
        far (N - points outside the box): the end of the ray inside the box
        mask (N): whether the ray intersects the box
    '''
    bounds = bounds + torch.tensor([boffset[0], boffset[1]])[:, None].to(device=orig.device)
    nominator = bounds[None] - orig[:, None] # N, 2, 3

    # calculate the step of intersections at six planes of the 3d bounding box
    direct = direct.detach().clone()
    direct[direct.abs() < 1e-5] = 1e-5
    d_intersect = (nominator / direct[:, None]).reshape(-1, 6)

    # calculate the six interections
    p_intersect = d_intersect[..., None] * direct[:, None] + orig[:, None]

    # calculate the intersections located at the 3d bounding box
    bounds = bounds.reshape(-1)
    min_x, min_y, min_z, max_x, max_y, max_z = bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))

    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(-1, 2, 3)

    # calculate the step of intersections
    norm_ray = torch.linalg.norm(direct[mask_at_box], dim=1)
    d0 = torch.linalg.norm(p_intervals[:, 0] - orig[mask_at_box], dim=1) / norm_ray
    d1 = torch.linalg.norm(p_intervals[:, 1] - orig[mask_at_box], dim=1) / norm_ray
    d01 = torch.stack((d0, d1), -1)
    near = d01.min(-1).values
    far = d01.max(-1).values
    return near, far, mask_at_box
