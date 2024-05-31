import numpy as np
from math import ceil
from numba import njit, jit, float64, uint32, int32
from typing import Tuple, List

"""
 MIT License

 Copyright (c) 2024 Carlos Cabaço Tojal

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

"""


def estimate_voxel_size(num_desired_voxels: int,
                        max_x: float, max_y: float, max_z: float,
                        min_x: float, min_y: float, min_z: float) -> Tuple[float, int, int, int, float, float, float]:
    """
    Estimate the voxel size for a given number of desired voxels, given space dimensions.

    Args:
        num_desired_voxels (int): Desired number of voxels.
        max_x (float): Maximum x coordinate.
        max_y (float): Maximum y coordinate.
        max_z (float): Maximum z coordinate.
    
    Returns:
        Tuple[float, int, int, int, float, float, float]: Tuple containing the voxel size, the number of voxels in each dimension, and the offsets in each dimension.
    """

    # calculate the lengths in each dimension
    x_dim = max_x - min_x
    y_dim = max_y - min_y
    z_dim = max_z - min_z

    # calculate the voxel size
    voxel_size = (x_dim * y_dim * z_dim) / num_desired_voxels

    # calculate the number of voxels in each dimension
    len_x = int(ceil(x_dim / voxel_size))
    len_y = int(ceil(y_dim / voxel_size))
    len_z = int(ceil(z_dim / voxel_size))

    # calculate the offsets in each dimension
    x_offset = min_x
    y_offset = min_y
    z_offset = min_z

    return voxel_size, len_x, len_y, len_z, x_offset, y_offset, z_offset