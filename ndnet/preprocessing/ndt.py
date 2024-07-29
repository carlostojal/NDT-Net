import numpy as np
from .core.ndt import ndt_downsample
from typing import Tuple

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


class NDT_Sampler:
    """
    Class for the Normal Distribution Transform (NDT) sampler.
    """

    def __init__(self, pointcloud: np.ndarray, classes: np.ndarray = None, num_classes: int = -1) -> None:
        """
        Create an NDT sampler.

        Args:
            pointcloud: The pointcloud.
            classes: The classes of the pointcloud.
            num_classes: The total number of classes.

        Returns:
            None
        """

        self.pointcloud: np.ndarray = pointcloud
        self.classes: np.ndarray = classes
        self.num_classes: int = num_classes

    def downsample(self, num_desired_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downsample the pointcloud.

        Args:
            num_desired_points: The number of desired points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The downsampled pointcloud, the downsampled classes
            and the downsampled covariances.
        """

        # downsample the pointcloud
        downsampled_pointcloud, downsampled_classes, downsampled_covariances = ndt_downsample(self.pointcloud,
                                                                                              num_desired_points,
                                                                                              self.classes, self.num_classes)

        return downsampled_pointcloud, downsampled_classes, downsampled_covariances
