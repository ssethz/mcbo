import pytest
from scripts.functions import Env, Dropwave, Alpine2, Ackley, Rosenbrock, ToyGraph, PSAGraph
import torch

def	test_function():
	f = Env()
	x = torch.tensor([0.5, 0.5])
	with pytest.raises(NotImplementedError):
		f.evaluate(x)

def test_dropwave():
	f = Dropwave(noise_scales=0.) 
	x = torch.tensor([0.5, 0.5])
	assert abs( f.evaluate(x)[-1] - 1.) < 0.0001

def test_alpine2():
	f = Alpine2(noise_scales=0.) 
	x = torch.tensor([0., 0., 0., 0., 0., 0.])
	assert abs( f.evaluate(x)[-1] - 0.) < 0.0001


def test_ackley():
	f = Ackley(noise_scales=0.) 
	# test that putting 0.5 for all actions give output 0
	x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	assert abs( f.evaluate(x)[-1] - 0.) < 0.0001
	with pytest.raises(ValueError):
		f.evaluate(torch.tensor([0.5, 0.5]))

def test_rosenbrock():
	f = Rosenbrock(noise_scales=0.) 
	x = torch.tensor([0.75, 0.75, 0.75, 0.75, 0.75])
	assert abs( f.evaluate(x)[-1] - 0.) < 0.0001

def test_toy():
	f = ToyGraph(noise_scales=0.) 
	#test that the max found in Aglietti has the right output value
	x = torch.tensor([0, 1, 0, 0., (-3.16053512 +5)/25, 0.])
	assert abs( f.evaluate(x)[-1] - (2.1710)) < 0.01
	x2 = torch.tensor([0.0000, 1.0000, 0.0000, 0.9109, 0.3227, 0.])
	assert abs( f.evaluate(x2)[-1] - (1.85562)) < 0.01

def test_PSA():
	f = PSAGraph() 
	x = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
	mean = 0
	n = 1000
	for _ in range(n):
		mean += f.evaluate(x)[-1] / n
	assert abs( mean - (-5.151712726)) < 0.05
