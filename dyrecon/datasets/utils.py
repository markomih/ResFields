import numpy as np
import cv2
import torch
from typing import List, Optional
from packaging import version
import torch.nn.functional as F

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose



def normalize_depth(depth_map, intrinsics):
    h,w = depth_map.shape
    directions = unproject_points(
        create_meshgrid(h, w, normalized_coordinates=False).reshape(-1, 2),
        torch.ones((h*w, 1)),
        torch.from_numpy(intrinsics).reshape(1, 3, 3)).reshape(h, w, 3)
    directions_norm = torch.norm(directions, dim=-1).numpy()
    depth_normalized = directions_norm*depth_map
    return depth_normalized

def denormalize_depth(depth_normalized, intrinsics):
    h,w = depth_normalized.shape
    directions = unproject_points(
        create_meshgrid(h, w, normalized_coordinates=False).reshape(-1, 2).to(depth_normalized),
        torch.ones((h*w, 1)).to(depth_normalized),
        intrinsics[:3, :3].reshape(1, 3, 3).to(depth_normalized)).reshape(h, w, 3)
    directions_norm = torch.norm(directions, dim=-1)#H,W,3
    depth_map = depth_normalized/directions_norm
    return depth_map


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    # TODO: torchscript doesn't like `torch_version_ge`
    # if torch_version_ge(1, 13, 0):
    #     x, y = torch_meshgrid([xs, ys], indexing="xy")
    #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
    # TODO: remove after we drop support of old versions
    if torch_version_ge(1, 10, 0):

        def torch_meshgrid(tensors: List[torch.Tensor], indexing: str):
            return torch.meshgrid(tensors, indexing=indexing)

    else:
        # TODO: remove this branch when kornia relies on torch >= 1.10.0
        def torch_meshgrid(tensors: List[torch.Tensor], indexing: str):
            return torch.meshgrid(tensors)
    base_grid: torch.Tensor = torch.stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def torch_version_ge(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch.__version__.split('+')[0])
    return _version >= version.parse(f"{major}.{minor}.{patch}")

def unproject_points(
    point_2d: torch.Tensor, depth: torch.Tensor, camera_matrix: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    r"""Unproject a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d: tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.

    Returns:
        tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)." " Got {}".format(depth.shape))

    xy: torch.Tensor = normalize_points_with_intrinsics(point_2d, camera_matrix)
    xyz: torch.Tensor = convert_points_to_homogeneous(xy)
    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2.0)

    return xyz * depth
def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return F.pad(points, [0, 1], "constant", 1.0)

def normalize_points_with_intrinsics(point_2d: torch.Tensor, camera_matrix: torch.Tensor):
    r"""Normalizes points with intrinsics. Useful for conversion of keypoints to be used with essential matrix.

    Args:
        point_2d: tensor containing the 2d points in the image pixel coordinates.
        The shape of the tensor can be :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 2)
        >>> K = torch.eye(3)[None]
        >>> normalize_points_with_intrinsics(X, K)
        tensor([[0.4963, 0.7682]])
    """
    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: torch.Tensor = point_2d[..., 0]
    v_coord: torch.Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xy: torch.Tensor = torch.stack([x_coord, y_coord], dim=-1)
    return xy
