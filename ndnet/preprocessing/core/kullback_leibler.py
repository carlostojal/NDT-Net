import numpy as np
from typing import List, Tuple
import bisect
from ndnet.preprocessing.core.normal_distributions import NormalDistribution

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

        if self.p.num_samples <= 1 or self.q.num_samples <= 1:
            raise RuntimeError(f"The number of samples in the distributions must be greater than 1!"
                             f" Had {self.p.num_samples} and {self.q.num_samples}.")
        
        # verify singular matrices
        if np.linalg.matrix_rank(self.p.covariance) < self.p.mean_.shape[0] or np.linalg.matrix_rank(self.q.covariance) < self.q.mean_.shape[0] or np.linalg.det(self.p.covariance) == 0 or np.linalg.det(self.q.covariance) == 0:
            raise RuntimeError(f"The covariance matrices of the distributions are singular!")

        mean_diff = self.q.mean_ - self.p.mean_
        q_cov_inv = np.linalg.inv(self.q.covariance)

        a = np.matmul(np.matmul(mean_diff.T, q_cov_inv), mean_diff)
        b = np.trace(np.matmul(q_cov_inv, self.p.covariance))
        c = np.log(np.linalg.det(self.p.covariance) / np.linalg.det(self.q.covariance))

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

                    divergence = None
                    div_value = 0.0
                    try:
                        divergence = KullbackLeiblerDivergence(current, neighbor)
                        div_value = divergence.divergence
                    except RuntimeError as e:
                        continue

                    # add the divergence to the list based on its "divergence" property
                    bisect.insort(divergences, divergence, key=lambda x: x.divergence)

    return divergences, num_valid_nds
