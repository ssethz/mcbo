import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from decimal import Decimal, ROUND_HALF_UP
import wandb
import math
import copy
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.utils.sampling import draw_sobol_samples


def inverse_sigmoid(x):
    return math.log(1.0 / (1.0 - x) - 1.0)


def vector_inverse_sigmoid(x):
    r"""
    Applies inverse_sigmoid to an entire vector elementwise.
    """
    return torch.log(1.0 / (1.0 - x) - 1.0)


def _no_grad_uniform_inverse_logistic_(tensor, a, b):
    r"""
    Fills tensor elementwise with elements from inverse_logistic(Uniform(a, b)).
    """
    with torch.no_grad():
        tensor.uniform_(a, b)
        return tensor.detach().apply_(inverse_sigmoid)


def uniform_inverse_logistic_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    r"""Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`. Then passes through inverse sigmoid.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_inverse_logistic(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            uniform_inverse_logistic_, (tensor,), tensor=tensor, a=a, b=b
        )
    return _no_grad_uniform_inverse_logistic_(tensor, a, b)


def round(x: float):
    return Decimal(x).quantize(0, ROUND_HALF_UP).to_integral_value()


class EtaNet(nn.Module):
    def __init__(self, input_dim, output_dim, weights=None):
        r"""
        Parameters
        ----------
        input_dim : int
        output_dim : int
        weights : int or None
            None if weights should be randomly initialized, else a vector of weights in
            [0, 1] should be specified.
        """
        super(EtaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4 * input_dim)
        self.fc2 = nn.Linear(4 * input_dim, output_dim)

        self.fc_baseline = nn.Linear(input_dim, output_dim)
        if weights is None:
            r"""
            Inverse logistic is used so that after being passed through a Sigmoid, the
            initial outputs of the baseline are uniform in [0.001, 0.999].
            """
            uniform_inverse_logistic_(self.fc_baseline.bias, 0.001, 0.999)
        else:
            with torch.no_grad():
                self.fc_baseline.bias.data.fill_(vector_inverse_sigmoid(weights))

    def forward(self, x):
        # betas are a constant (that can be uniformly randomized) plus an adaptive term
        baseline = self.fc_baseline(torch.zeros(x.shape))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x + baseline)  # to constrain outputs to 0,1


class ActionNet(nn.Module):
    def __init__(self, input_dim, output_dim, weights=None):
        r"""
        Parameters
        ----------
        input_dim : int
        output_dim : int
        weights : int or None
            None if weights should be randomly initialized, else a vector of weights in
            [0, 1] should be specified.
        """
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)
        if weights is None:
            r"""
            Inverse logistic is used so that after being passed through a Sigmoid, the
            initial actions are uniform in [0.001, 0.999].
            """
            uniform_inverse_logistic_(self.fc1.weight, 0.001, 0.999)
        else:
            with torch.no_grad():
                self.fc1.weight.data = vector_inverse_sigmoid(weights.unsqueeze(dim=1))

    def forward(self, x):
        y = self.fc1(x)
        return torch.sigmoid(y)  # [0,1] output


def get_nets(env_profile, num_actions, obs_dim=1):
    r"""
    Initializes networks for each eta (including constant for each root eta) and actions
    .
    Parameters
    ----------
    env_profile : Dict
    num_actions: int
        The input dimensionality. Number of actions in the final action vector.

    Returns
    -------
    eta_list : nn.ModuleDict
        A dictionary holding the action network and a list of eta networks (one for each
        node).
    """
    dag = env_profile["dag"]
    root_nodes = dag.get_root_nodes()
    n_nodes = dag.get_n_nodes()
    eta_nets = []
    for k in range(n_nodes):
        parent_nodes = dag.get_parent_nodes(k)
        parent_actions = env_profile["active_input_indices"][k]
        action_length = len(parent_actions)
        if len(parent_actions) == 0 and len(parent_nodes) == 0:
            # if no parents then add a dummy variable for eta to take as input
            action_length = action_length + 1
        t = EtaNet(len(parent_nodes) * obs_dim + action_length, obs_dim, weights=None)
        eta_nets.append(t)
    return nn.ModuleDict(
        [
            ["actions", ActionNet(1, num_actions, weights=None)],
            ["etas", nn.ModuleList(eta_nets)],
        ]
    )


def train(
    model,
    network_to_objective_transform,
    env_profile,
    input_dim,
    old_nets,
    epochs=1000,
    batch_size=32,
    lr=0.001,
    reinits=1000,
):
    r"""
    Selects action and eta_net to optimize the upper confidence bound objective.
    Uses random reinits of action and eta networks combined with some gradient descent
    steps. Also uses the best previously used solutions as initializations.

    Parameters
    ----------
    model : GaussianProcessNetwork
        GP Network fit to the current data.
    network_to_objective_transform : Tensor -> Tensor
        Takes samples of all system variables and returns just the reward variable
        samples.
    env_profile : dict
    input_dim : int
        The number of action variables.
    old_nets : List[Dict]
        A list of the ActionNet and EtaNet objects for all previous actions taken.
    epochs : int
        Max number of rounds for gradient descent to take before stopping.
    batch_size : int
        Number of repeats we average across when estimating the objective for each
        gradient descent iteration.
    lr : float
    reinits : int
        Number of times to randomly reinitialize networks weights when searching over
        action and eta nets.

    Output
    ------
    best_sol : Tensor
        The set of actions given by the best solution found.
    best_sol_score : Tensor
        The objective value obtained by the best solution found.
    best_nets : Dict
        The action and eta nets of the best solution.
    """
    best_sol_score = -torch.inf
    best_sol = None
    best_nets = None

    for j in range(reinits + len(old_nets)):
        # If used all old inits we start generating random new ones.
        if j >= len(old_nets):
            nets = get_nets(env_profile, input_dim)
        else:
            nets = copy.deepcopy(old_nets[j])

        posterior = model.noisy_posterior(nets)
        optimizer = torch.optim.SGD(nets.parameters(), lr=lr, maximize=True)
        stop = False
        stopping_criterion = ExpMAStoppingCriterion(maxiter=epochs)
        while not stop:
            optimizer.zero_grad()
            objective_value = torch.mean(
                network_to_objective_transform(
                    posterior.rsample(sample_shape=torch.Size([batch_size]))
                )
            )
            objective_value.backward()
            optimizer.step()
            stop = stopping_criterion.evaluate(fvals=objective_value.detach())
        with torch.no_grad():
            r"""
            Compute the objective with a lot of samples to get a low variance
            performance estimate.
            """
            num_samples = 1000
            objective_value = torch.mean(
                network_to_objective_transform(
                    posterior.rsample(sample_shape=torch.Size([num_samples]))
                )
            )
        if objective_value >= best_sol_score:
            with torch.no_grad():
                best_sol = nets["actions"].forward(torch.ones(1, 1))
                best_sol_score = objective_value
                best_nets = nets
        optimizer.zero_grad()
    wandb.log({"acq_value": best_sol_score}, commit=False)
    return best_sol, best_sol_score, best_nets
