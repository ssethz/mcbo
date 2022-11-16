import os
import torch
import botorch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import wandb
import argparse

import warnings

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
debug._set_state(True)

from mcbo.mcbo_trial import mcbo_trial
from mcbo.utils import runner_utils

n_bo_iter = 100

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument(
    "-a", "--algo", type=str, default="Random", help="Which algorithm to use"
)
parser.add_argument(
    "-b",
    "--beta",
    type=runner_utils.check_nonnegative,
    default=5.0,
    help="Value of beta in UCB algorithms",
)
parser.add_argument(
    "-e", "--env", type=str, default="Dropwave", help="Environment to evaluate on"
)
parser.add_argument(
    "-n",
    "--noise_scale",
    type=runner_utils.check_nonnegative,
    default=0.0,
    help="Fixed noise scale of every node",
)
parser.add_argument(
    "--scratch",
    type=bool,
    default=True,
    help="True if saving wandb logs to a scratch folder",
)
parser.add_argument(
    "--initial_obs_samples",
    type=int,
    default=10,
    choices=range(1, 100),
    help="If causal, number of observational samples.",
)
parser.add_argument(
    "--initial_int_samples",
    type=int,
    default=2,
    choices=range(1, 100),
    help="If causal, number of initial interventional samples per node.",
)
parser.add_argument(
    "--batch_size", type=int, default=32, choices=range(1, 100), help="Batch size of NMCBO in noisy settings"
)
args = parser.parse_args()

# if saving to scratch, tell wandb to save there
if args.scratch:
    wandb.init(
        project="mbcbo",
        dir=os.environ.get("SCRATCH"),
        entity="ADD WANDB ENTITY HERE",
    )
else:
    wandb.init(project="mbcbo", entity="ADD WANDB ENTITY HERE")

wandb.config.update(args)

noise_scale = wandb.config["noise_scale"]

from functions import Dropwave, Alpine2, Ackley, Rosenbrock, ToyGraph, PSAGraph
if wandb.config["env"] == "Dropwave":
    env = Dropwave(noise_scales=noise_scale)
elif wandb.config["env"] in ["Alpine2", "Alpine"]:
    env = Alpine2(noise_scales=noise_scale)
elif wandb.config["env"] == "Ackley":
    env = Ackley(noise_scales=noise_scale)
elif wandb.config["env"] == "Rosenbrock":
    env = Rosenbrock(noise_scales=noise_scale)
elif wandb.config["env"] == "ToyGraph":
    env = ToyGraph(noise_scales=noise_scale)
elif wandb.config["env"] == "PSAGraph":
    env = PSAGraph()
else:
    raise ValueError("Invalid environment specified")


def function_network(X: Tensor):
    return env.evaluate(X=X)


# Function that maps the network output to the objective value
network_to_objective_transform = lambda Y: Y[..., -1]
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
if noise_scale > 0.001:
    batch_size = wandb.config["batch_size"]
else:
    batch_size = 2
env_profile = env.get_env_profile()
algo_profile = {
    "algo": wandb.config["algo"],
    "seed": wandb.config["seed"],
    "n_init_evals": 2 * (env_profile["input_dim"] + 1),
    "n_bo_iter": n_bo_iter,
    "beta": wandb.config["beta"],
    "initial_obs_samples": wandb.config["initial_obs_samples"],
    "initial_int_samples": wandb.config["initial_int_samples"],
    "batch_size": batch_size,
}

mcbo_trial(
    algo_profile=algo_profile,
    env_profile=env_profile,
    function_network=function_network,
    network_to_objective_transform=network_to_objective_transform,
)
