import pytest 
import torch
import argparse
from mbcbo.utils import functions_utils, runner_utils

def test_noise_scales_to_normals():
	from torch.distributions.normal import Normal

	# test scalar noise_scales
	distribution_list = functions_utils.noise_scales_to_normals(2.0, 2)
	for d in distribution_list:
		assert d.variance == 2.0**2

	# test a torch tensor of noise scales
	distribution_list = functions_utils.noise_scales_to_normals(torch.ones((2,)), 2)
	for d in distribution_list:
		assert d.variance == 1.0

	#test zero
	distribution_list = functions_utils.noise_scales_to_normals(0.0, 2)
	for d in distribution_list:
		assert d.variance == 1e-6

def test_check_nonnegative():
	with pytest.raises(argparse.ArgumentTypeError):
		runner_utils.check_nonnegative(-2)
	with pytest.raises(argparse.ArgumentTypeError):
		runner_utils.check_nonnegative('hello world')
	assert runner_utils.check_nonnegative(0.0) == 0.0
	assert runner_utils.check_nonnegative(2) == 2.0