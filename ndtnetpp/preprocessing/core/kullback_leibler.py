import numpy as np
from typing import List, Tuple
import bisect
from ndtnetpp.preprocessing.core.normal_distributions import NormalDistribution

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
        self.div_val = None

    @property
    def divergence(self) -> float:
        # only compute the divergence the first time it is requested
        if self.div_val is None:
            self.compute()
        return self.div_val

    def compute(self) -> float:
        a = np.transpose(self.q.mean_ - self.p.mean_) * np.linalg.inv(self.q.covariance) * (self.q.mean_ - self.p.mean_)
        b = np.trace(np.linalg.inv(self.q.covariance) * self.p.covariance)
        c = np.log(np.linalg.det(self.q.covariance) / np.linalg.det(self.p.covariance))

        self.div_val = 0.5 * (a + b + c - self.p.mean_.shape[0])

        return self.div_val


def calculate_kl_divergences(grid: np.ndarray[NormalDistribution],
                             lens: np.ndarray) -> Tuple[List[KullbackLeiblerDivergence], int]:
    """
    Calculate the Kullback-Leibler divergence between neighboring distributions in the grid.

    Args:
        grid (np.ndarray): A grid of normal distributions.
        lens (np.ndarray): The size of the grid in each dimension.

    Returns:
        Tuple[List[KullbackLeiblerDivergence], int]: The divergences and the number of valid normal distributions.
    """

    divergences: List[KullbackLeiblerDivergence] = []
    num_valid_nds: int = 0

    # iterate the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                # get the current distribution
                current = grid[i, j, k]

                if current.num_samples == 0:
                    continue

                num_valid_nds += 1

                # get the neighbors
                neighbors = []
                if i > 0:
                    neighbors.append(grid[i - 1, j, k])
                if i < lens[0] - 1:
                    neighbors.append(grid[i + 1, j, k])
                if j > 0:
                    neighbors.append(grid[i, j - 1, k])
                if j < lens[1] - 1:
                    neighbors.append(grid[i, j + 1, k])
                if k > 0:
                    neighbors.append(grid[i, j, k - 1])
                if k < lens[2] - 1:
                    neighbors.append(grid[i, j, k + 1])

                # calculate the Kullback-Leibler divergence
                for neighbor in neighbors:
                    divergence = KullbackLeiblerDivergence(current, neighbor)

                    # add the divergence to the list based on its "divergence" property
                    bisect.insort(divergences, divergence, key=lambda x: x.divergence)

    return divergences, num_valid_nds
