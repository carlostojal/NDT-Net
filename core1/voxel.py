import numpy as np
from math import ceil, floor
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
                        max_limits: np.ndarray,
                        min_limits: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimate the voxel size for a given number of desired voxels, given space dimensions.

    Args:
        num_desired_voxels (int): Desired number of voxels.
        max_limits (ndarray): Maximum limits of the coordinates in each dimension.
        min_limits (ndarray): Minimum limits of the coordinates in each dimension.
    
    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Tuple containing the voxel size and the number of voxels in each dimension.
    """

    # calculate the lengths
    dims = max_limits - min_limits

    # calculate the voxel size
    voxel_size: float = np.prod(dims) / num_desired_voxels

    # calculate the number of voxels in each dimension
    lens: np.ndarray = np.ceil(dims/voxel_size, dtype=int)

    return voxel_size, lens


def estimate_voxel_grid(max_limits: np.ndarray,
                        min_limits: np.ndarray,
                        voxel_size: float) -> np.ndarray:
    """
    Estimate the voxel grid for a given voxel size, given space dimensions.

    Args:
        max_limits (ndarray): Maximum limits of the coordinates in each dimension.
        min_limits (ndarray): Minimum limits of the coordinates in each dimension.
        voxel_size (float): Voxel size.

    Returns:
        ndarray: Voxel grid dimensions.
    """

    # calculate the lengths
    dims = max_limits - min_limits

    # calculate the number of voxels in each dimension
    lens = np.ceil(dims/voxel_size).astype(int)

    return lens

def metric_to_voxel_space(point: np.ndarray, voxel_size: float,
                          lens: np.ndarray,
                          min_limits: np.ndarray) -> np.ndarray:
    """
    Convert a point from metric space to voxel space.

    Args:
        point (ndarray): Point coordinates in metric space.
        voxel_size (flaot): Voxel size.
        lens (ndarray): Voxel grid dimensions.
        min_limits (ndarray): Minimum limits of the coordinates in each dimension.

    Returns:
        ndarray: Point coordinates in voxel space.
    """
    
    # calculate the voxel indexes
    indexes = np.floor((point - min_limits) / voxel_size).astype(int)

    if np.any(indexes < 0) or np.any(indexes >= lens):
        raise ValueError("Point is outside the voxel grid.")

    return indexes
