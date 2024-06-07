import numpy as np
from typing import List, Tuple
from ndtnetpp.preprocessing.core.normal_distributions import NormalDistribution, estimate_ndt
from ndtnetpp.preprocessing.core.kullback_leibler import KullbackLeiblerDivergence, calculate_kl_divergences
from ndtnetpp.preprocessing.core.voxel import estimate_voxel_grid

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

DOWNSAMPLE_UPPER_THRESHOLD = 0.2
MIN_VOXEL_SIZE = 0.01
MAX_VOXEL_SIZE = 10.0
MAX_GUESS_ITERATIONS = 15

def prune_nds(nds: np.ndarray[NormalDistribution],
              num_valid_nds: int,
              num_desired_nds: int,
              kl_divergences: List[KullbackLeiblerDivergence]) -> np.ndarray[NormalDistribution]:
    """
    Prune normal distributions based on their likelihoods.

    Args:
        nds: The normal distributions.
        num_desired_nds: The number of normal distributions to keep.
        kl_divergences: The Kullback-Leibler divergences between the normal distributions.

    Returns:
        The pruned normal distributions.
    """

    if num_desired_nds > num_valid_nds:
        raise ValueError("The number of desired normal distributions is too large!")

    to_remove: int = num_valid_nds - num_desired_nds
    removed: int = 0

    while removed < to_remove and len(kl_divergences) > 0:

        # this normal distribution has no samples or was already removed
        if kl_divergences[0].p.num_samples == 0:
            # remove the Kullback-Leibler divergence from the list
            kl_divergences.pop(0)
            continue

        # remove the normal distribution
        kl_divergences[0].p.num_samples = 0

        # remove the Kullback-Leibler divergence from the list
        kl_divergences.pop(0)

        removed += 1

    return nds


def to_point_cloud(nds: np.ndarray[NormalDistribution]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Convert normal distributions to a point cloud.

    Args:
        nds: The normal distributions.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]: The point cloud, the covariances and the classes.
    """

    points: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    classes: List[np.ndarray] = []

    # iterate the grid
    for i in range(nds.shape[0]):
        for j in range(nds.shape[1]):
            for k in range(nds.shape[2]):

                # verify if the distribution has samples
                if nds[i, j, k].num_samples == 0:
                    continue

                # get the point
                point = nds[i, j, k].mean_
                points.append(point)

                # get the covariance
                covariance = nds[i, j, k].covariance
                covariances.append(covariance)

                cls = nds[i, j, k].class_tag
                if cls == -1:
                    cls = 0
                # create a one-hot encoding of the class
                cls1 = np.zeros(nds[i, j, k].num_classes+1)
                cls1[cls] = 1
                classes.append(cls1)

    return np.array(points, dtype=float), np.array(covariances, dtype=float), np.array(classes, dtype=float)


def ndt_downsample(pointcloud: np.ndarray,
                   num_desired_points: int,
                   classes: np.ndarray = None,
                   num_classes: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample a point cloud using the Normal Distribution Transform (NDT).

    Args:
        pointcloud: The point cloud.
        num_desired_points: The number of desired points.
        classes: The classes of the points.
        num_classes: The total number of classes.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: The downsampled point cloud, the covariances and the classes.
    """

    # get the limits of the point cloud
    min_limits = np.min(pointcloud, axis=0)
    max_limits = np.max(pointcloud, axis=0)

    # guess the voxel size
    guess: float = (MAX_VOXEL_SIZE - MIN_VOXEL_SIZE) / 2.0
    min_guess = MIN_VOXEL_SIZE
    max_guess = MAX_VOXEL_SIZE

    nds = None
    lens = None

    it: int = 0
    while it <= MAX_GUESS_ITERATIONS:

        # estimate the voxel grid size
        lens = estimate_voxel_grid(max_limits, min_limits, guess)

        # estimate the normal distribution transform
        nds, num_valid_nds = estimate_ndt(pointcloud, guess, lens, min_limits, classes, num_classes)

        # adjust the voxel size using binary search, until the desired number of points is reached
        if num_valid_nds > num_desired_points * (1 + DOWNSAMPLE_UPPER_THRESHOLD):
            min_guess = guess
        elif num_valid_nds < num_desired_points:
            max_guess = guess
        else:
            break
        # update the guess
        guess = (min_guess + max_guess) / 2.0

        it += 1

    # reached the maximum number of iterations
    if it == MAX_GUESS_ITERATIONS:
        raise ValueError("The maximum number of iterations was reached!")

    # calculate the Kullback-Leibler divergences
    kl_divergences, num_valid_nds = calculate_kl_divergences(nds, lens)

    # prune the normal distributions
    nds = prune_nds(nds, num_valid_nds, num_desired_points, kl_divergences)

    # convert the normal distributions to a point cloud
    points, covariances, classes = to_point_cloud(nds)

    return points, covariances.reshape((num_desired_points, 9)), classes
