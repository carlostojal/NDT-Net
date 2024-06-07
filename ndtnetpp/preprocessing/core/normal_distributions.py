import numpy as np
from typing import Tuple
from threading import Thread, Lock, Condition
from ndtnetpp.preprocessing.core.voxel import metric_to_voxel_space

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


class NormalDistribution:
    """
    A Normal Distribution.
    """
    def __init__(self, index: np.ndarray, num_classes: int = -1) -> None:
        """
        Create a Normal Distribution.

        Args:
            index: The index of the distribution.
            num_classes: The total number of classes.
        """
        self.index: np.ndarray = index
        self.mean_: np.ndarray = np.zeros(3, dtype=np.float64)
        self.old_mean: np.ndarray = np.zeros(3, dtype=np.float64)
        self.covariance: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.m2: np.ndarray = np.zeros(3, dtype=np.float64)  # sum of squared differences
        self.num_samples: int = 0
        self.num_classes: int = num_classes
        self.class_tag: int = -1
        self.class_samples = None
        if num_classes != -1:
            self.class_samples: np.ndarray = np.zeros(num_classes+1, dtype=np.int32)  # number of samples per class
        self.lock = Lock()  # mutex lock
        self.cv = Condition(self.lock)  # condition variable
        self.being_updated = False

    def update(self, sample: np.ndarray, class_tag: int = -1) -> None:
        """
        Update the normal distribution with a new sample.

        Args:
            sample: The new sample.
            class_tag: The class tag of the sample.

        Returns:
            None
        """

        # set the distribution as being updated
        self.being_updated = True

        # increment the sample count
        self.num_samples += 1

        # copy the old mean
        self.old_mean = self.mean_.copy()

        # update the mean and covariance
        self.mean_ += (sample - self.mean_) / self.num_samples
        self.m2 += (sample - self.mean_) * (sample - self.old_mean)
        # TODO: update the variance and covariance
        for i in range(3):
            for j in range(3):
                # the diagonal elements are the variance
                if i == j:
                    self.covariance[i, j] = self.m2[i] / self.num_samples
                else:
                    # the off-diagonal elements are the covariance
                    self.covariance[i, j] += (sample[i] - self.old_mean[i]) * (sample[j] - self.old_mean[j]) / self.num_samples


        # update the class tag
        if class_tag != -1:
            # update the class samples
            self.class_samples[class_tag] += 1

            # update the class tag
            self.class_tag = np.argmax(self.class_samples)

        # set the distribution as not being updated
        self.being_updated = False


def estimate_ndt(pointcloud: np.ndarray,
                 voxel_size: float,
                 lens: np.ndarray,
                 min_limits: np.ndarray,
                 classes: np.ndarray = None,
                 num_classes: int = -1,
                 num_workers: int = 8) -> Tuple[np.ndarray, int]:
    """
    Estimate the Normal Distribution Transform (NDT) of a point cloud for a given voxel size.

    Args:
        pointcloud: The point cloud.
        classes: The classes of the points.
        num_classes: The total number of classes.
        voxel_size: The voxel size.
        lens: Number of voxels in each dimension.
        min_limits: The minimum limits of the point cloud.
        num_workers: The number of worker threads.

    Returns:
        Tuple[np.ndarray, int]: The NDT grid and the number of valid normal distributions.
    """

    # create the grid
    grid = np.empty(lens, dtype=object)
    for index in np.ndindex(*lens):
        grid[index] = NormalDistribution(np.array(index), num_classes)

    num_valid_nds = 0

    # iterate the point cloud
    for idx in range(pointcloud.shape[0]):
        # get the voxel index
        voxel_index = metric_to_voxel_space(pointcloud[idx], voxel_size, lens, min_limits)

        # get the class tag
        class_tag = -1
        if classes is not None:
            class_tag = classes[idx]

        nd: NormalDistribution = grid[tuple(voxel_index)]

        # if the distribution had no previous samples, increment the number of valid normal distributions
        if nd.num_samples == 0:
            num_valid_nds += 1

        # update the normal distribution
        nd.update(pointcloud[idx], class_tag)

    return grid, num_valid_nds
