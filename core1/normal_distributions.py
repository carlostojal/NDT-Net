import numpy as np

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
    def __init__(self, index: np.ndarray, num_classes: int = -1):
        self.index = index
        self.mean = np.zeros(3, dtype=np.float64)
        self.old_mean = np.zeros(3, dtype=np.float64)
        self.covariance = np.zeros((3, 3), dtype=np.float64)
        self.m2 = np.zeros(3, dtype=np.float64)  # sum of squared differences
        self.num_samples: int = 0
        self.num_classes = num_classes
        self.class_tag: int = -1
        self.class_samples = None
        if num_classes != -1:
            self.class_samples = np.zeros(num_classes, dtype=np.int32)  # number of samples per class

    def update(self, sample: np.ndarray, class_tag: int = -1):
        # increment the sample count
        self.num_samples += 1

        # copy the old mean
        self.old_mean = self.mean.copy()

        # update the mean and covariance
        self.mean += (sample - self.mean) / self.num_samples
        self.m2 += (sample - self.mean) * (sample - self.old_mean)
        self.covariance = self.m2 / self.num_samples

        # update the class tag
        if class_tag != -1:
            # update the class samples
            self.class_samples[class_tag] += 1

            # update the class tag
            self.class_tag = np.argmax(self.class_samples)
