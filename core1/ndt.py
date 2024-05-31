import numpy as np
from typing import List, Tuple
from core1.normal_distributions import NormalDistribution
from core1.kullback_leibler import KullbackLeiblerDivergence, calculate_kl_divergences
from core1.voxel import voxel_to_metric_space

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
            kl_divergences.pop()

        # remove the normal distribution
        kl_divergences[0].p.num_samples = 0

        # remove the Kullback-Leibler divergence from the list
        kl_divergences.pop()

        removed += 1

    return nds


def to_point_cloud(nds: np.ndarray[NormalDistribution],
                   lens: np.ndarray[int],
                   min_limits: np.ndarray[float],
                   voxel_size: float) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Convert normal distributions to a point cloud.

    Args:
        nds: The normal distributions.
        lens: The number of samples per normal distribution.
        min_limits: The minimum limits of the point cloud.
        voxel_size: The size of the voxels.

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
                cls1 = np.zeros(nds[i, j, k].num_classes)
                cls1[cls] = 1
                classes.append(cls1)

    return np.array(points, dtype=float), np.array(covariances, dtype=float), np.array(classes, dtype=float)
