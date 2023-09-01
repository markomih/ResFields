import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional
import numpy as np
import math
from models.utils import masked_mean
from scipy import signal

class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction='mean'):
        assert reduction in ['none', 'sum', 'mean', 'valid_mean']
        loss = self.func(inputs, targets, reduction='none')
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'valid_mean':
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = F.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        elif reduction == 'sum':
            return _ssim.sum()
        elif reduction == 'mean':
            return _ssim.mean()


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

@torch.jit.script
def compute_dist_loss(pred_weights: torch.Tensor, svals: torch.Tensor) -> torch.Tensor:
    """Compute the distortion loss of each ray.

    Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.
        Barron et al., CVPR 2022.
        https://arxiv.org/abs/2111.12077

    As per Equation (15) in the paper. Note that we slightly modify the loss to
    account for "sampling at infinity" when rendering NeRF.

    Args:
        pred_weights (jnp.ndarray): (..., S, 1) predicted weights of each
            sample along the ray.
        svals (jnp.ndarray): (..., S + 1, 1) normalized marching step of each
            sample along the ray.
    """
    pred_weights = pred_weights[..., 0]

    # (..., S)
    smids = 0.5 * (svals[..., 1:, 0] + svals[..., :-1, 0])
    sdeltas = svals[..., 1:, 0] - svals[..., :-1, 0]

    loss1 = (pred_weights[..., None, :]*pred_weights[..., None]*torch.abs(smids[..., None, :] - smids[..., None])).sum(dim=(-2, -1))
    loss2 = 1 / 3 * (pred_weights**2 * sdeltas).sum(dim=-1)

    loss = loss1 + loss2
    return loss.mean()

# @torch.jit.script
def compute_depth_loss(depth: torch.Tensor, pred_depth: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(pred_depth, depth, reduction='none')
    loss = masked_mean(loss, (depth > 0).float())
    return loss


def compute_psnr(img0: torch.Tensor, img1: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.

    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / math.log(10)*torch.log(masked_mean(mse, mask))

def sparse_loss(sdf_rnd, sdf_ray):
    # sparse loss from SparseNeuS
    sparse_loss_1 = torch.exp(-1000 * torch.abs(sdf_ray)).mean()
    sparse_loss_2 = torch.exp(-100 * torch.abs(sdf_rnd)).mean()
    sparse_loss = (sparse_loss_1+sparse_loss_2)*0.5
    return sparse_loss

def compute_ssim(
    # img0: jnp.ndarray,
    img0: torch.Tensor,
    # img1: jnp.ndarray,
    img1: torch.Tensor,
    # mask: Optional[jnp.ndarray] = None,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
# ) -> jnp.ndarray:
) -> torch.Tensor:
    """Computes SSIM between two images.

    This function was modeled after tf.image.ssim, and should produce
    comparable output.

    Image Inpainting for Irregular Holes Using Partial Convolutions.
        Liu et al., ECCV 2018.
        https://arxiv.org/abs/1804.07723

    Note that the mask operation is implemented as partial convolution. See
    Section 3.1.

    Args:
        img0 (jnp.ndarray): An image of size (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of size (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.
        max_val (float): The dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size (int): Size of the Gaussian blur kernel used to smooth the
            input images.
        filter_sigma (float): Standard deviation of the Gaussian blur kernel
            used to smooth the input images.
        k1 (float): One of the SSIM dampening parameters.
        k2 (float): One of the SSIM dampening parameters.

    Returns:
        jnp.ndarray: SSIM in range [0, 1] of shape ().
    """

    img0 = torch.as_tensor(img0).detach().cpu()
    img1 = torch.as_tensor(img1).detach().cpu()
    

    if mask is None:
        # mask = jnp.ones_like(img0[..., :1])
        mask = torch.ones_like(img0[..., :1])
    mask = mask[..., 0]  # type: ignore

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    # f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    f_i = ((torch.arange(filter_size).cpu() - hw + shift) / filter_sigma) ** 2
    # filt = jnp.exp(-0.5 * f_i)
    filt = torch.exp(-0.5 * f_i)
    # filt /= jnp.sum(filt)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # NOTICE Dusan: previous version used vectorization on Color channel, we need to avoid this
    def convolve2d(z, m, f):
        z_ = []
        for i in range(3):
            z_.append(torch.as_tensor(signal.convolve2d(z[...,i] * m, f, mode="valid")).cpu())
        z_ = torch.stack(z_, axis=-1)

        m_ = torch.as_tensor(signal.convolve2d(m, torch.ones_like(f), mode="valid")).cpu()

        return_where = []
        for i in range(3):
            return_where.append(torch.where(m_ != 0, z_[...,i] * torch.ones_like(f).sum() / m_, torch.tensor(0., device='cpu')))
        
        return_where = torch.stack(return_where, axis=-1)

        return return_where, (m_ != 0).type(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
    filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(img0, mask)[0]
    mu1 = filt_fn(img1, mask)[0]
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, mask)[0] - mu00
    sigma11 = filt_fn(img1**2, mask)[0] - mu11
    sigma01 = filt_fn(img0 * img1, mask)[0] - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    # sigma00 = jnp.maximum(0.0, sigma00)
    sigma00 = torch.maximum(torch.tensor(0.0).cpu(), sigma00)
    # sigma11 = jnp.maximum(0.0, sigma11)
    sigma11 = torch.maximum(torch.tensor(0.0).cpu(), sigma11)
    # sigma01 = jnp.sign(sigma01) * jnp.minimum(
        # jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
    # )
    sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = ssim_map.mean()

    return ssim

def get_compute_lpips() -> Callable[
    # [np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
]:
    """Get the LPIPS metric function.

    Note that torch and jax does not play well together. This means that
    running them in the same process on GPUs will cause issue.

    A workaround for now is to run torch on CPU only. For LPIPS computation,
    the overhead is not too bad.
    """

    import lpips
    import torch

    model = lpips.LPIPS(net="alex", spatial=True)

    @torch.inference_mode()
    def compute_lpips(
        # img0: np.ndarray, img1: np.ndarray, mask: Optional[np.ndarray] = None
    # ) -> np.array:
        img0: torch.Tensor, img1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute LPIPS between two images.

        This function computes mean LPIPS over masked regions. The input images
        are also masked. The following previous works leverage this metric:

        [1] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic
        Scenes.
            Li et al., CVPR 2021.
            https://arxiv.org/abs/2011.13084

        [2] Transforming and Projecting Images into Class-conditional
        Generative Networks.
            Huh et al., CVPR 2020.
            https://arxiv.org/abs/2005.01703

        [3] Controlling Perceptual Factors in Neural Style Transfer.
            Gatys et al., CVPR 2017.
            https://arxiv.org/abs/1611.07865

        Args:
            img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            mask (Optional[jnp.ndarray]): An optional forground mask of shape
                (H, W, 1) in float32 {0, 1}. The metric is computed only on the
                pixels with mask == 1.

        Returns:
            np.ndarray: LPIPS in range [0, 1] in shape ().
        """
        if mask is None:
            mask = np.ones_like(img0[..., :1])
        img0 = lpips.im2tensor(np.array(img0 * mask), factor=1 / 2)
        img1 = lpips.im2tensor(np.array(img1 * mask), factor=1 / 2)
        return masked_mean(model(img0, img1).cpu().numpy()[0, 0, ..., None], mask)

    return compute_lpips
