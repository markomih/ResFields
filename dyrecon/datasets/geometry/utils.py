#!/usr/bin/env python3
#
# File   : utils.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# import jax.numpy as jnp
import torch
import numpy as np


def matmul(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
    else:
        # assert isinstance(a, jnp.ndarray)
        assert isinstance(b, torch.Tensor)
        # assert isinstance(b, jnp.ndarray)
        assert isinstance(a, torch.Tensor)

    if isinstance(a, np.ndarray):
        return a @ b
    else:
        # NOTE: The original implementation uses highest precision for TPU
        # computation. Since we are using GPUs only, comment it out.
        #  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
        # return jnp.matmul(a, b)
        return torch.matmul(a, b)


def matv(a, b):
    return matmul(a, b[..., None])[..., 0]


@torch.no_grad()
def tringulate_rays(origins, viewdirs) -> np.ndarray:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (types.Array): A (N, 3) array of ray origins.
        viewdirs (types.Array): A (N, 3) array of ray view directions.

    Returns:
        np.ndarray: A (3,) lookat point.
    """
    # tf.config.set_visible_devices([], "GPU")

    origins = np.array(origins[None], np.float32)
    viewdirs = np.array(viewdirs[None], np.float32)
    weights = np.ones(origins.shape[:2], dtype=np.float32)
    points = np.array(torch_ray_triangulate(origins, origins + viewdirs, weights))
    return points[0]


def assert_all_above(
    # vector: type_alias.TensorLike,
    vector : torch.Tensor,
    # minval: type_alias.TensorLike,
    minval : torch.Tensor,
    open_bound: bool = False,) -> torch.Tensor:
    """Checks whether all values of vector are above minval.
    Note:
        In the following, A1 to An are optional batch dimensions.
    Args:
        vector: A tensor of shape `[A1, ..., An]` containing the values we want to
        check.
        minval: A scalar or a tensor of shape `[A1, ..., An]` representing the
        desired lower bound for the values in `vector`.
        open_bound: A `bool` indicating whether the range is open or closed.
        name: A name for this op. Defaults to 'assert_all_above'.
    Raises:
        tf.errors.InvalidArgumentError: If any entry of the input is below `minval`.
    Returns:
        The input vector, with dependence on the assertion operator in the graph.
    """


    # vector = tf.convert_to_tensor(value=vector)
    vector = torch.tensor(vector)
    # minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
    minval = torch.tensor(minval, dtype=vector.dtype)

    if open_bound:
    #   assert_ops = (tf.debugging.assert_greater(vector, minval),)
        assert_ops = (torch.gt(vector, minval),)
    else:
    #   assert_ops = (tf.debugging.assert_greater_equal(vector, minval),)
        assert_ops = (torch.ge(vector, minval),)

    assert all(assert_ops[0].squeeze(0))

    with torch.no_grad():
        return vector

def assert_at_least_k_non_zero_entries(
    # tensor: type_alias.TensorLike,
    tensor : torch.Tensor,
    k: int = 1,
    # name: str = 'assert_at_least_k_non_zero_entries') -> tf.Tensor:
    ) -> torch.Tensor:

    """Checks if `tensor` has at least k non-zero entries in the last dimension.
    Given a tensor with `M` dimensions in its last axis, this function checks
    whether at least `k` out of `M` dimensions are non-zero.
    Note:
        In the following, A1 to An are optional batch dimensions.
    Args:
        tensor: A tensor of shape `[A1, ..., An, M]`.
        k: An integer, corresponding to the minimum number of non-zero entries.
        name: A name for this op. Defaults to 'assert_at_least_k_non_zero_entries'.
    Raises:
        InvalidArgumentError: If `tensor` has less than `k` non-zero entries in its
        last axis.
    Returns:
        The input tensor, with dependence on the assertion operator in the graph.
    """
    # tensor = tf.convert_to_tensor(value=tensor)
    tensor = torch.tensor(tensor)

    # indicator = tf.cast(tf.math.greater(tensor, 0.0), dtype=tensor.dtype)
    indicator = torch.gt(tensor, 0.0).type(tensor.dtype)

    # indicator_sum = tf.reduce_sum(input_tensor=indicator, axis=-1)
    indicator_sum = torch.sum(indicator, dim=-1)
    # assert_op = tf.debugging.assert_greater_equal(
        # indicator_sum, tf.cast(k, dtype=tensor.dtype))
    assert_op = torch.ge(indicator_sum, torch.tensor(k, dtype=tensor.dtype))
    # with tf.control_dependencies([assert_op]):
    assert all(assert_op)
    # return tf.identity(tensor)

    with torch.no_grad():
        return tensor

def torch_ray_triangulate(startpoints, endpoints, weights, name="ray_triangulate"):
    """
    Copied from tensorflow_graphics.geometry.representation.ray.ray_triangulate
    Triangulates 3d points by miminizing the sum of squared distances to rays.

    The rays are defined by their start points and endpoints. At least two rays
    are required to triangulate any given point. Contrary to the standard
    reprojection-error metric, the sum of squared distances to rays can be
    minimized in a closed form.

    Note:
        In the following, A1 to An are optional batch dimensions.

    Args:
        startpoints: A tensor of ray start points with shape `[A1, ..., An, V, 3]`,
        the number of rays V around which the solution points live should be
        greater or equal to 2, otherwise triangulation is impossible.
        endpoints: A tensor of ray endpoints with shape `[A1, ..., An, V, 3]`, the
        number of rays V around which the solution points live should be greater
        or equal to 2, otherwise triangulation is impossible. The `endpoints`
        tensor should have the same shape as the `startpoints` tensor.
        weights: A tensor of ray weights (certainties) with shape `[A1, ..., An,
        V]`. Weights should have all positive entries. Weight should have at least
        two non-zero entries for each point (at least two rays should have
        certainties > 0).
        name: A name for this op. The default value of None means "ray_triangulate".

    Returns:
        A tensor of triangulated points with shape `[A1, ..., An, 3]`.

    Raises:
        ValueError: If the shape of the arguments is not supported.
    """
    # startpoints = tf.convert_to_tensor(value=startpoints)
    startpoints = torch.tensor(startpoints)
    # endpoints = tf.convert_to_tensor(value=endpoints)
    endpoints = torch.tensor(endpoints)
    # weights = tf.convert_to_tensor(value=weights)
    weights = torch.tensor(weights)


    # weights = asserts.assert_all_above(weights, 0.0, open_bound=False)
    weights = assert_all_above(weights, 0.0, open_bound=False)
    # weights = asserts.assert_at_least_k_non_zero_entries(weights, k=2)
    weights = assert_at_least_k_non_zero_entries(weights, k=2)
    

    left_hand_side_list = []
    right_hand_side_list = []
    # TODO(b/130892100): Replace the inefficient for loop and add comments here.
    for ray_id in range(weights.shape[-1]):
        weights_single_ray = weights[..., ray_id]
        startpoints_single_ray = startpoints[..., ray_id, :]
        endpoints_singleview = endpoints[..., ray_id, :]
        ray = endpoints_singleview - startpoints_single_ray
        # ray = tf.nn.l2_normalize(ray, axis=-1)
        ray = torch.nn.functional.normalize(ray, dim=-1)
        # ray_x, ray_y, ray_z = tf.unstack(ray, axis=-1)
        ray_x, ray_y, ray_z = torch.unbind(ray, dim=-1)
        # zeros = tf.zeros_like(ray_x)
        zeros = torch.zeros_like(ray_x)
        # cross_product_matrix = tf.stack(
            # (zeros, -ray_z, ray_y, ray_z, zeros, -ray_x, -ray_y, ray_x, zeros),
            # axis=-1)
        cross_product_matrix = torch.stack(
            (zeros, -ray_z, ray_y, ray_z, zeros, -ray_x, -ray_y, ray_x, zeros),
            dim=-1)
        # cross_product_matrix_shape = tf.concat(
            # (tf.shape(input=cross_product_matrix)[:-1], (3, 3)), axis=-1)
        cross_product_matrix_shape = torch.cat(
            (torch.tensor(cross_product_matrix.shape[:-1]), torch.tensor((3, 3))), dim=-1)
        # cross_product_matrix = tf.reshape(
            # cross_product_matrix, shape=cross_product_matrix_shape)
        cross_product_matrix = torch.reshape(
            cross_product_matrix, shape=tuple(cross_product_matrix_shape.numpy()))
        # weights_single_ray = tf.expand_dims(weights_single_ray, axis=-1)
        weights_single_ray = torch.unsqueeze(weights_single_ray, dim=-1)
        # weights_single_ray = tf.expand_dims(weights_single_ray, axis=-1)
        weights_single_ray = torch.unsqueeze(weights_single_ray, dim=-1)
        left_hand_side = weights_single_ray * cross_product_matrix
        left_hand_side_list.append(left_hand_side)
        # dot_product = tf.matmul(cross_product_matrix,
                                # tf.expand_dims(startpoints_single_ray, axis=-1))
        dot_product = torch.matmul(cross_product_matrix,    
                                torch.unsqueeze(startpoints_single_ray, dim=-1))
        right_hand_side = weights_single_ray * dot_product
        right_hand_side_list.append(right_hand_side)
        # left_hand_side_multi_rays = tf.concat(left_hand_side_list, axis=-2)
        left_hand_side_multi_rays = torch.cat(left_hand_side_list, dim=-2)
        # right_hand_side_multi_rays = tf.concat(right_hand_side_list, axis=-2)
        right_hand_side_multi_rays = torch.cat(right_hand_side_list, dim=-2)
        # points = tf.linalg.lstsq(left_hand_side_multi_rays,
                                # right_hand_side_multi_rays)
        points = torch.linalg.lstsq(left_hand_side_multi_rays,
                                right_hand_side_multi_rays).solution
        # points = tf.squeeze(points, axis=-1)
        points = torch.squeeze(points, dim=-1)

    return points
