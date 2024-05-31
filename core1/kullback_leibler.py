import numpy as np
from core1.normal_distributions import NormalDistribution

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


class KullbackLeiblerDivergence:
    def __init__(self, p: NormalDistribution, q: NormalDistribution) -> None:
        self.p = p
        self.q = q
        self.computed = False

    @property
    def divergence(self) -> float:
        return self.compute()

    def compute(self) -> float:
        a = np.transpose(self.q.mean - self.p.mean) * np.linalg.inv(self.q.covariance) * (self.q.mean - self.p.mean)
        b = np.trace(np.linalg.inv(self.q.covariance) * self.p.covariance)
        c = np.log(np.linalg.det(self.q.covariance) / np.linalg.det(self.p.covariance))

        return 0.5 * (a + b + c - self.p.mean.shape[0])
