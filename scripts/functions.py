r"""
Defines the environments that BO agents take actions in. 

References:

.. [astudilloBayesianOptimizationFunction2021]
    Astudillo, Raul, and Peter Frazier. "Bayesian optimization of function networks." 
    Advances in Neural Information Processing Systems 34 (2021): 14463-14475.
.. [agliettiCausalBayesianOptimization2020]
    Aglietti, Virginia, et al. "Causal bayesian optimization." International Conference
    on Artificial Intelligence and Statistics. PMLR, 2020.
"""

import torch
import math
import numpy as np

from mcbo.utils.dag import DAG, FNActionInput
from mcbo.utils import functions_utils
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

class Env:
    """
    Abstract class for environments.
    """

    def evaluate(self, X):
        r"""
        For action input X returns the observation of all network nodes.
        """
        raise NotImplementedError

    def check_input(self, X):
        """
        Checks if the input to evaluate matches the correct input dim. 
        """
        if X.shape[-1] != self.input_dim:
            raise ValueError("Input dimension does not fit for this function")

    def get_causal_quantities(self):
        """
        Outputs a dict of environment details that are computed differently for 
        CausalEnv and FNEnv. 
        """
        raise NotImplementedError

    def get_base_profile(self):
        """
        Outputs a dict of environment details that can be computed similarly for all
        types of environments. 
        """
        return {
            "additive_noise_dists": self.additive_noise_dists,
            "interventional": self.interventional,
            "dag": self.dag,
            "input_dim": self.input_dim,
        }

    def get_env_profile(self):
        """
        Outputs a dictionary containing all details of the environment needed for 
        BO experiments. 
        """
        return {**self.get_base_profile(), **self.get_causal_quantities()}


class FNEnv(Env):
    """
    Abstract class for function network environments. All function network environments
    implemented as subclasses first appeared in 
    [astudilloBayesianOptimizationFunction2021].
    """

    def __init__(self, dag: DAG, action_input: FNActionInput):
        self.dag = dag
        self.input_dim = action_input.get_input_dim()
        self.action_input = action_input
        self.interventional = False

    def get_causal_quantities(self):
        # no targets because in FNEnv actions are added nodes, not do-interventions.
        valid_targets = [torch.zeros(self.input_dim)]
        do_map = None
        active_input_indices = self.action_input.get_active_input_indices()
        return {
            "valid_targets": valid_targets,
            "do_map": do_map,
            "active_input_indices": active_input_indices,
        }


class CausalEnv(Env):
    """
    Abstract class for environments based upon do() interventions. All CausalEnvs 
    implemented as subclasses first appeared in 
    [agliettiCausalBayesianOptimization2020].
    """

    def __init__(self, dag: DAG):
        self.dag = dag
        # Every node has a binary (whether to intervene) and an intervention value.
        self.input_dim = dag.get_n_nodes() * 2
        self.interventional = True

    def do_int(self, unintervened_value_i, do_x_i, x_i):
        r"""
        Given a batch of unintervened_value_i, do_x_i and x_i computes a batch of 
        observed variable values. 
        Parameters
        ----------
        unintervened_value_i : Tensor
            Node value if not intervened upon.
        do_x_i : Tensor
            For each element in the batch, 1 if the node is intervened upon, 0 
            otherwise.
        x_i : Tensor
            For each element in the batch, the node value if it was intervened upon. 
        
        Return
        ------
        t : Tensor
            The batch of node values after considering if interventions were performed. 
        """
        return unintervened_value_i * (1 - do_x_i) + do_x_i * x_i

    def do_map(self, X):
        r"""Map the inputs from [0,1] to the space of intervention inputs. Required 
        because all our BO algorithms always select actions in [0,1]."""
        raise NotImplementedError

    def get_causal_quantities(self):
        # for CausalEnv we use no active input indices on any nodes
        active_input_indices = [list() for _ in range(self.input_dim)]
        valid_targets = self.valid_targets
        do_map = self.do_map
        return {
            "valid_targets": valid_targets,
            "do_map": do_map,
            "active_input_indices": active_input_indices,
        }


class Dropwave(FNEnv):
    """
    A modification of the classic Drop-Wave test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [0]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1], []]
        action_input = FNActionInput(active_input_indices)
        super(Dropwave, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 10.24 * X - 5.12
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        norm_X = torch.norm(X_scaled, dim=-1)
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = norm_X + noise_0
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (1.0 + torch.cos(12.0 * norm_X)) / (
            2.0 + 0.5 * (norm_X**2)
        ) + noise_1
        return output


class Alpine2(FNEnv):
    """
    A modification of the classic Alpine test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [0], [1], [2], [3], [4]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0], [1], [2], [3], [4], [5]]
        action_input = FNActionInput(active_input_indices)
        super(Alpine2, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 10.0 * X
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        for i in range(self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            if i == 0:
                noise_0 = self.additive_noise_dists[0].rsample(
                    sample_shape=output[..., 0].shape
                )
                output[..., i] = -torch.sqrt(x_i) * torch.sin(x_i) + noise_0
            else:
                noise_i = self.additive_noise_dists[i].rsample(
                    sample_shape=output[..., i].shape
                )
                output[..., i] = (
                    torch.sqrt(x_i) * torch.sin(x_i) * output[..., i - 1] + noise_i
                )
        return output


class Ackley(FNEnv):
    """
    A modification of the classic Ackley test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [], [0, 1]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []]
        action_input = FNActionInput(active_input_indices)
        super(Ackley, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 4.0 * (X - 0.5)
        output = torch.empty(X_scaled.shape[:-1] + (self.dag.get_n_nodes(),))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            1 / self.input_dim * torch.sum(torch.square(X_scaled[..., :]), dim=-1)
            + noise_0
        )
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (
            1
            / self.input_dim
            * torch.sum(torch.cos(2 * math.pi * X_scaled[..., :]), dim=-1)
            + noise_1
        )
        noise_2 = self.additive_noise_dists[2].rsample(
            sample_shape=output[..., 2].shape
        )
        output[..., 2] = (
            20 * torch.exp(-0.2 * torch.sqrt(output[..., 0]))
            + torch.exp(output[..., 1])
            - 20
            - math.e
            + noise_2
        )
        return output


class Rosenbrock(FNEnv):
    """
    A modification of the classic Rosenbrock test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [0], [1], [2]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1], [1, 2], [2, 3], [3, 4]]
        action_input = FNActionInput(active_input_indices)
        super(Rosenbrock, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 4.0 * (X - 0.5)
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            -100 * torch.square(X_scaled[..., 1] - torch.square(X_scaled[..., 0]))
            - torch.square(1 - X_scaled[..., 0])
            + noise_0
        )
        for i in range(1, self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            j = i + 1
            x_j = X_scaled[..., j]
            noise_i = self.additive_noise_dists[i].rsample(
                sample_shape=output[..., i].shape
            )
            output[..., i] = (
                -100 * torch.square(x_j - torch.square(x_i))
                - torch.square(1 - x_i)
                + output[..., i - 1]
                + noise_i
            )

        return output


class ToyGraph(CausalEnv):
    r""" The ToyGraph environment from [agliettiCausalBayesianOptimization2020].
    """
    def __init__(self, noise_scales=1.0):
        parent_nodes = [[], [0], [1]]
        dag = DAG(parent_nodes)
        super(ToyGraph, self).__init__(dag)
        self.valid_targets = [
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 0]),
        ]

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

    def do_map(self, X):
        r"""Maps an input in [0,1] to the range used in this environment"""
        B = X.clone()
        B[..., 0] = 10.0 * X[..., 0] - 5.0
        B[..., 1] = 25 * X[..., 1] - 5.0
        return B

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[
            ..., : self.dag.get_n_nodes()
        ]  # seperate the binary part from the intervention part
        X = X[..., self.dag.get_n_nodes() :]
        X = self.do_map(X)
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise0 = self.additive_noise_dists[0].rsample(sample_shape=output[..., 0].shape)
        output[..., 0] = self.do_int(noise0, X_do[..., 0], X[..., 0])

        noise1 = self.additive_noise_dists[1].rsample(sample_shape=output[..., 1].shape)
        output[..., 1] = self.do_int(
            (torch.exp(-output[..., 0]) + noise1), X_do[..., 1], X[..., 1]
        )

        noise2 = self.additive_noise_dists[2].rsample(sample_shape=output[..., 2].shape)
        unintervened_output_2 = (
            torch.cos(output[..., 1]) - torch.exp(-output[..., 1] / 20.0) + noise2
        )
        r'''
        [agliettiCausalBayesianOptimization2020] always minimizes outcomes whilst we 
        maximize, so I take the negative of the final output. 
        '''
        output[..., 2] = -self.do_int(unintervened_output_2, X_do[..., 2], X[..., 2])
        return output


class PSAGraph(CausalEnv):
    r""" The PSAGraph environment from [agliettiCausalBayesianOptimization2020].
    """
    def __init__(self):
        # ordering: age, bmi, A, S, cancer, Y
        parent_nodes = [[], [0], [0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
        dag = DAG(parent_nodes)
        super(PSAGraph, self).__init__(dag)
        self.valid_targets = [
            torch.tensor([0, 0, 1, 0, 0, 0]),
            torch.tensor([0, 0, 0, 1, 0, 0]),
            torch.tensor([0, 0, 1, 1, 0, 0]),
        ]
        self.additive_noise_dists = [
            Uniform(-10, 10),
            Normal(0, 0.7),
            Normal(0, 0.001),
            Normal(0, 0.001),
            Normal(0, 0.001),
            Normal(0, 0.4),
        ]

    def do_map(self, X):
        return X

    def f_age(self, shape):
        # easier to just sample implicitely. Only use the additive_noise_dists
        # for passing to the experiments module
        return functions_utils.uniform(shape, 55, 75)

    def f_bmi(self, age):
        return torch.normal(27.0 - 0.01 * age, 0.7)

    def f_A(self, age, bmi):
        return torch.sigmoid(-8.0 + 0.10 * age + 0.03 * bmi)

    def f_S(self, age, bmi):
        return torch.sigmoid(-13.0 + 0.10 * age + 0.20 * bmi)

    def f_cancer(self, age, bmi, A, S):
        return torch.sigmoid(2.2 - 0.05 * age + 0.01 * bmi - 0.04 * S + 0.02 * A)

    def f_Y(self, age, bmi, A, S, cancer):
        return torch.normal(
            6.8 + 0.04 * age - 0.15 * bmi - 0.60 * S + 0.55 * A + 1.00 * cancer, 0.4
        )

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[
            ..., : self.dag.get_n_nodes()
        ]  # seperate the binary part from the intervention part
        X = X[..., self.dag.get_n_nodes() :]
        X = self.do_map(X)
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        output[..., 0] = self.do_int(
            self.f_age(output[..., 0].shape), X_do[..., 0], X[..., 0]
        )
        output[..., 1] = self.do_int(
            self.f_bmi(output[..., 0]), X_do[..., 1], X[..., 1]
        )
        output[..., 2] = self.do_int(
            self.f_A(output[..., 0], output[..., 1]), X_do[..., 2], X[..., 2]
        )

        output[..., 3] = self.do_int(
            self.f_S(output[..., 0], output[..., 1]), X_do[..., 3], X[..., 3]
        )

        unintervened_output_4 = self.f_cancer(
            output[..., 0], output[..., 1], output[..., 2], output[..., 3]
        )
        output[..., 4] = self.do_int(unintervened_output_4, X_do[..., 4], X[..., 4])
        r'''
        [agliettiCausalBayesianOptimization2020] always minimizes outcomes whilst we 
        maximize, so I take the negative of the final output. 
        '''
        output[..., 5] = -self.f_Y(
            output[..., 0],
            output[..., 1],
            output[..., 2],
            output[..., 3],
            output[..., 4],
        )
        return output
