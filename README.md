# mcbo
The code companion to the paper "Model-based Causal Bayesian Optimization". 

## Credit
The starting point for the code in this repository was
https://github.com/RaulAstudillo06/BOFN. 

We build on BOTorch (https://botorch.org/). 

## Conda Environment
In a new conda environment with Python 3.9 run
```conda install botorch -c pytorch -c gpytorch -c conda-forge```
Then in the base directory of this repository:
```pip install -e .``` 

On your system you now have a conda environment called "mcbo".
This should be loaded whenever you run experiments.

## Running
You can launch experiments by running `scripts/runner.py` and controlling the command line inputs.
All experimental results are logged to the Weights and Bias service. 

## Naming
`MCBO` is the algorithm studied in the Model-based Causal Bayesian Optimization paper.
The algorithm in this repo named `MCBO` is designed for just near-noiseless environments
(like Function Networks). The algorithm named `NMCBO` implements `MCBO` for noisy
environments. 

## File Structure
`mcbo` provides the core functionality of model-based causal bayesian optimization. 
In this folder, 
`mcbo_trial.py` implements the environment interaction loop. 
`models/gp_network.py` contains the class for fitting GPs for EIFN and MCBO
`models/eta_network.py` contains the training loop for the custom optimizer used for
optimizing the acquisition function in `NMCBO`. All other methods use default BOTorch
optimizers. 

`scripts` provides the key functionality for running experiments. 

