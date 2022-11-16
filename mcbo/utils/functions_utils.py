r"""
A set of helper functions for the functions.py script.
"""
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution


class ZeroDist(Distribution):
    r"""Creates a torch Distribution that is always 0."""

    # has a negligible nonzero variance for stability in downstrream tasks
    variance = 1e-6
    arg_constraints = {}

    def rsample(self, sample_shape=torch.Size([])):
        return torch.zeros(sample_shape)


def uniform(shape, low, high):
    r"""Elementwise uniform dist from low to high"""
    return torch.rand(shape) * (high - low) + low


def noise_scales_to_normals(noise_scales, n_nodes: int):
    r"""
    Creates a list of additive Gaussian noise Distributions (one for each node) based on
    specified noise scales.
    Parameters
    ----------
    noise_scales : float or Tensor
        Specifies the variance scale for the additive noise of each node. Tensor must be
        of shape (n_nodes). If a single float is used, it is applied as the variance
        scale to every Distribution.
    n_nodes : int
        The number of nodes in the graph.

    Returns
    -------
    l : List[Normal]
        A list of PyTorch Normal distributions with mean 0.0 and variance specified by
        the noise_scales.
    """
    if isinstance(noise_scales, float):
        return [
            ZeroDist() if noise_scales <= 1e-6 else Normal(0, noise_scales)
            for _ in range(n_nodes)
        ]
    elif isinstance(noise_scales, torch.Tensor):
        if noise_scales.shape[0] == n_nodes:
            return [
                ZeroDist() if noise_scales[i] <= 1e-6 else Normal(0, noise_scales[i])
                for i in range(noise_scales.shape[0])
            ]
        else:
            raise ValueError("noise_scales is tensor but not of length n_nodes")
    raise ValueError("noise_scales must be float or tensor")
