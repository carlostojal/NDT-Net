import numpy as np
from typing import List
from core1.normal_distributions import NormalDistribution
from core1.kullback_leibler import KullbackLeiblerDivergence, calculate_kl_divergences

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
              lens: np.ndarray,
              num_desired_nds: int,
              kl_divergences: List[KullbackLeiblerDivergence]) -> np.ndarray[NormalDistribution]:
    """
    Prune normal distributions based on their likelihoods.

    Args:
        nds: The normal distributions.
        lens: The likelihoods.
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
