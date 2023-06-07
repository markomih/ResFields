import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import itertools

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('eq', lambda a, b: a == b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
OmegaConf.register_new_resolver('instant_ngp_scale', lambda n_levels, base_res, max_res: float(np.exp(np.log(max_res / base_res) / (n_levels - 1))))
OmegaConf.register_new_resolver('torch', lambda cmd: eval(f'torch.{cmd}'))
# ======================================================= #

def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)

def plot_heatmap(data: np.ndarray, min_val=-3, max_val=3, resize_ratio: int=2):
    if len(data.shape) == 2:
        data = data[..., None]

    data = data.clip(min_val, max_val)
    normalized_data = (data - min_val) / (max_val-min_val)  # [0, 1]
    
    img = np.uint8(normalized_data * 255)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    img = cv2.resize(img, dsize=(img.shape[1]*resize_ratio, img.shape[0]*resize_ratio))
    return img

def load_config(yaml_file, cli_args=[]):
    yaml_conf = OmegaConf.load(yaml_file)
    if yaml_conf.get('defaults', False):
        dir_name = os.path.dirname(yaml_file)
        defaults = [OmegaConf.load(os.path.join(dir_name, f)) for f in yaml_conf.defaults]
        defaults = OmegaConf.merge(*defaults)
        yaml_conf = OmegaConf.merge(defaults, yaml_conf)
    
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    OmegaConf.resolve(conf)
    return conf

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def feat_sample(feat, uv, mode='bilinear', padding_mode='zeros', align_corners=True):
    '''
    args:
        feat: (B, C, H, W)
        uv: (B, N, 2) [-1, 1]
    return:
        (B, N, C)
    '''
    uv = uv[:, :, None]
    feat = torch.nn.functional.grid_sample(
        feat,
        uv,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return feat.view(*feat.shape[:2], -1).permute(0, 2, 1)

def create_tri_planes(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, w).expand(1, h, w)  # [1, H, W]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, h, 1).expand(1, h, w)  # [1, H, W]
    xy_plane = torch.cat((x_range, y_range), dim=0).reshape(1, 2,-1).permute(0,2,1)

    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h).expand(1, d, h)  # [1, D, H]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1).expand(1, d, h)  # [1, D, H]
    yz_plane = torch.cat((y_range, z_range), dim=0).reshape(1, 2,-1).permute(0,2,1)

    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, 1, d).expand(1, w, d)  # [1, W, D]
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, w, 1).expand(1, w, d)  # [1, W, D]
    zx_plane = torch.cat((z_range, x_range), dim=0).reshape(1, 2,-1).permute(0,2,1)

    return xy_plane, yz_plane, zx_plane

def create_tri_time_planes(d, h, w, t, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w, device=device)).view(1, 1, w).expand(1, t, w)  # [1, T, W]
    t_range = (torch.linspace(-1,1,steps=t, device=device)).view(1, t, 1).expand(1, t, w)  # [1, T, W]
    xt_plane = torch.cat((x_range, t_range), dim=0).reshape(1, 2, -1).permute(0, 2, 1)

    y_range = (torch.linspace(-1,1,steps=h, device=device)).view(1, 1, h).expand(1, t, h)  # [1, T, H]
    t_range = (torch.linspace(-1,1,steps=t, device=device)).view(1, t, 1).expand(1, t, h)  # [1, T, H]
    yt_plane = torch.cat((y_range, t_range), dim=0).reshape(1, 2,-1).permute(0,2,1)

    z_range = (torch.linspace(-1,1,steps=d, device=device)).view(1, 1, d).expand(1, t, d)  # [1, T, D]
    t_range = (torch.linspace(-1,1,steps=t, device=device)).view(1, t, 1).expand(1, t, d)  # [1, T, D]
    zt_plane = torch.cat((z_range, t_range), dim=0).reshape(1, 2,-1).permute(0,2,1)

    return xt_plane, yt_plane, zt_plane
