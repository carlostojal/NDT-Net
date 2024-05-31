import numpy as np
from typing import Tuple
from enum import Enum

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


class Direction(Enum):
    """
    Enumeration of the possible directions.
    """
    X_POS=0,
    X_NEG=1,
    Y_POS=2,
    Y_NEG=3,
    Z_POS=4,
    Z_NEG=5,
    DIRECTION_LEN=6


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


def voxel_to_metric_space(voxel: np.ndarray,
                          lens: np.ndarray,
                          min_limits: np.ndarray,
                          voxel_size: float) -> np.ndarray:
    """
    Convert a voxel from voxel space to metric space.

    Args:
        voxel (ndarray): Voxel coordinates in voxel space.
        lens (ndarray): Voxel grid dimensions.
        min_limits (ndarray): Minimum limits of the coordinates in each dimension.
        voxel_size (flaot): Voxel size.

    Returns:
        ndarray: Voxel coordinates in metric space.
    """

    point: np.ndarray = (voxel * voxel_size) + (voxel_size / 2) + min_limits

    return point


def voxel_pos_to_index(voxel_pos: np.ndarray, lens: np.ndarray) -> int:
    """
    Convert a voxel position to an index.

    Args:
        voxel_pos (ndarray): Voxel position.
        lens (ndarray): Voxel grid dimensions.

    Returns:
        int: Voxel index.
    """

    # check bounds
    if np.any(voxel_pos < 0) or np.any(voxel_pos >= lens):
        raise ValueError("Invalid position.")

    # calculate the index
    index: int = int(voxel_pos[2] * lens[0] * lens[1] + voxel_pos[1] * lens[0] + voxel_pos[0])

    return index


def index_to_voxel_pos(index: int, lens: np.ndarray) -> np.ndarray:
    """
    Convert a voxel index to a position.

    Args:
        index (int): Voxel index.
        lens (ndarray): Voxel grid dimensions.

    Returns:
        ndarray: Voxel position.
    """

    # check bounds
    if index < 0 or index >= np.prod(lens):
        raise ValueError("Invalid index.")

    # calculate the position
    pos: np.ndarray = np.zeros(3, dtype=int)
    pos[0] = index % lens[0]
    pos[1] = (index % (lens[0] * lens[1])) // lens[0]
    pos[2] = index // (lens[0] * lens[1])

    return pos


def get_neighbor_index(index: int,
                       lens: np.ndarray,
                       direction: Direction) -> int:
    """
    Get the index of the neighbor voxel in a given direction.

    Args:
        index (int): Index of the current voxel.
        lens (ndarray): Voxel grid dimensions.
        direction (Direction): Direction of the neighbor voxel.

    Returns:
        int: Index of the neighbor voxel.
    """

    # verify the index bounds
    if index < 0 or index >= np.prod(lens):
        raise ValueError("Invalid index.")

    # get the direction
    direction_vec: np.ndarray = np.zeros(3, dtype=int)
    match direction:
        case Direction.X_POS:
            direction_vec[0] = 1
        case Direction.X_NEG:
            direction_vec[0] = -1
        case Direction.Y_POS:
            direction_vec[1] = 1
        case Direction.Y_NEG:
            direction_vec[1] = -1
        case Direction.Z_POS:
            direction_vec[2] = 1
        case Direction.Z_NEG:
            direction_vec[2] = -1

    # update the index
    cur_pos: np.ndarray = index_to_voxel_pos(index, lens)
    cur_pos += direction_vec

    # check bounds
    if np.any(cur_pos < 0) or np.any(cur_pos >= lens):
        raise IndexError("Invalid neighbor.")

    # calculate the new index
    new_index: int = voxel_pos_to_index(cur_pos, lens)

    return new_index
