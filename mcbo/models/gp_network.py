#! /usr/bin/env python3

r"""
Gaussian Process Network and classes for selecting actions in a GP Network for various 
acquisition functions.
"""

from __future__ import annotations
import torch
from typing import Any
from botorch.models.model import Model
from botorch.models import FixedNoiseGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


def identity(x):
    return x


def create_constant_feature_vector(batch_shape):
    """
    Creates a constant feature vector of size batch_shape + (1, ). Used when fitting a
    GP to a variable that has no causes, since in BoTorch fitting a GP expects a
    nonempty feature vector.
    """
    return torch.zeros(batch_shape + (1,))


def check_no_inputs(k: int, is_root: bool, active_input_indices: List[List[int]]):
    r"""
    Returns true if a random variable has no causal parents (either actions or
    other variables in the causal graph).
    """
    return is_root and len(active_input_indices[k]) == 0


class GaussianProcessNetwork(Model):
    r"""Fits a GP model to the provided data given the graph structure."""

    def __init__(self, train_X, train_Y, algo_profile, env_profile) -> None:
        r""" """
        self.train_X = train_X
        self.train_Y = train_Y
        self.algo_profile = algo_profile
        self.env_profile = env_profile
        self.dag = env_profile["dag"]
        n_nodes = self.dag.get_n_nodes()

        # A list of GP models for each node.
        self.node_GPs = [None for k in range(n_nodes)]
        self.normalization_constant_lower = [
            [None for j in range(len(self.dag.get_parent_nodes(k)))]
            for k in range(n_nodes)
        ]
        self.normalization_constant_upper = [
            [None for j in range(len(self.dag.get_parent_nodes(k)))]
            for k in range(n_nodes)
        ]

        self.set_target(env_profile["valid_targets"][0])

        for k in self.env_profile["dag"].get_root_nodes():
            self._fit_single_node(k, True)

        for k in range(n_nodes):
            if self.node_GPs[k] is None:
                self._fit_single_node(k, False)

    @staticmethod
    def _check_no_inputs(k: int, is_root: bool, active_input_indices: List[List[int]]):
        r"""
        Returns true if a random variable has no causal parents (either actions or
        other variables in the causal graph).
        """
        return is_root and len(active_input_indices[k]) == 0

    @staticmethod
    def _create_intervention_mask(train_X_k: Tensor, interventional: bool):
        r"""
        For interventional data, we mask out all datapoints where the node was
        intervened upon
        """
        if interventional:
            r"""
            If we are considering hard interventions, the k'th entry in train_X is 0
            if we didn't intervene, and 1 otherwise.
            """
            mask = train_X_k.int() == 0
        else:
            mask = torch.zeros(train_X_k.shape) == 0
            assert torch.all(mask)
        return mask

    def _construct_features_k(self, k: int, is_root: bool):
        r"""
        Uses actions X to construct the feature vector for the GP at node k.
        """
        active_input_indices = self.env_profile["active_input_indices"]
        dag = self.env_profile["dag"]
        if check_no_inputs(k, is_root, active_input_indices):
            train_X_node_k = create_constant_feature_vector(self.train_X.shape[:-1])
        elif is_root:
            train_X_node_k = self.train_X[..., active_input_indices[k]]
        else:
            r"""
            If k is not a root, need to include the value of node k's parents as
            features.
            """
            aux = self.train_Y[..., dag.get_parent_nodes(k)].clone()
            for j in range(len(dag.get_parent_nodes(k))):
                self.normalization_constant_lower[k][j] = torch.min(aux[..., j])
                self.normalization_constant_upper[k][j] = torch.max(aux[..., j])
                aux[..., j] = (
                    aux[..., j] - self.normalization_constant_lower[k][j]
                ) / (
                    self.normalization_constant_upper[k][j]
                    - self.normalization_constant_lower[k][j]
                )
            train_X_node_k = torch.cat(
                [self.train_X[..., active_input_indices[k]], aux], -1
            )
        return train_X_node_k

    def _fit_single_node(self, k: int, is_root: bool):
        r"""
        Fits a GP for a single node k.
        """

        mask = self._create_intervention_mask(
            self.train_X[..., k], self.env_profile["interventional"]
        )

        train_X_node_k = self._construct_features_k(k, is_root)
        train_Y_node_k = self.train_Y[..., [k]]
        r""" 
        Since do-interventions will break the dependance on all parents, we remove data
        where a do-intervention was performed using mask. 
        """

        self.node_GPs[k] = FixedNoiseGP(
            train_X=train_X_node_k[mask],
            train_Y=train_Y_node_k[mask],
            train_Yvar=torch.ones(train_Y_node_k.shape)[mask]
            * self.env_profile["additive_noise_dists"][k].variance,
            outcome_transform=Standardize(m=1),
        )
        node_mlls = ExactMarginalLogLikelihood(
            self.node_GPs[k].likelihood, self.node_GPs[k]
        )
        fit_gpytorch_model(node_mlls)
        return None

    def set_target(self, target):
        """
        Sets the binary target variable so that noisy_posterior and posterior will
        compute the posterior for the case where do-interventions are performed on all
        indices with value 1 in target.
        Parameters
        ----------
        target : (n_nodes, ) Tensor
            A binary vector where 1 at the ith indicates a do() intervention on the ith
            variable.
        """
        self.target = target
        return None

    def noisy_posterior(self, nets) -> MultivariateNormalNetwork:
        """
        Return a NoisyHallucinatedGaussianProcessNetwork class which includes actions
        for controllling the realization of the epistemic uncertainty.
        Parameters
        ----------
        nets : Dict
            A dictionary containing ActionNets and EtaNets, which parameterize policies
            for selecting actions and controlling epistemic uncertainty respectively.

        Returns
        -------
        m : MultivariateNormalNetwork
            A NoisyHallucinatedGaussianProcessNetwork with the fit model and nets.
        """
        return NoisyHallucinatedGaussianProcessNetwork(
            self.node_GPs,
            nets,
            self.algo_profile,
            self.env_profile,
            (self.normalization_constant_lower, self.normalization_constant_upper),
            self.target,
        )

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
    ) -> MultivariateNormalNetwork:
        r"""Computes the posterior over model outputs at the provided points. If MCBO
        is specified as the algorithm, includes actions that control the epistemic
        uncertainty. Otherwise, returns a posterior as used in EIFN.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        if self.algo_profile["algo"] == "MCBO":
            return HallucinatedGaussianProcessNetwork(
                self.node_GPs,
                X,
                self.algo_profile,
                self.env_profile,
                (self.normalization_constant_lower, self.normalization_constant_upper),
                self.target,
            )
        return MultivariateNormalNetwork(
            self.node_GPs,
            X,
            self.algo_profile,
            self.env_profile,
            (self.normalization_constant_lower, self.normalization_constant_upper),
            self.target,
        )

    def forward(self, x: Tensor) -> MultivariateNormalNetwork:
        r"""
        Not needed for our experiments but included in the base class.
        """
        return NotImplementedError

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""
        Not needed for our experiments but included in the base class.
        """
        return NotImplementedError


class MultivariateNormalNetwork(Posterior):
    """
    The class used by EIFN to select the best action according to the EI acquisition
    function. Epistemic uncertainty is sampled from when forward-simulating the outcomes
    of actions.
    """

    def __init__(
        self, node_GPs, X, algo_profile, env_profile, normalization_constants, target
    ):
        self.algo_profile = algo_profile
        self.env_profile = env_profile
        self.node_GPs = node_GPs
        self.X = X

        self.normalization_constant_lower = normalization_constants[0]
        self.normalization_constant_upper = normalization_constants[1]
        self.target = target
        self.intervention_input_map()

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = self.env_profile["dag"].get_n_nodes()
        shape = torch.Size(shape)
        return shape

    def _create_nodes_sample_tensor(self, sample_shape):
        r"""
        Creates a Tensor of the right size for nodes_samples. This contains the batch of
        simulated samples for each index.
        """

        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [
            False for k in range(self.env_profile["dag"].get_n_nodes())
        ]
        return nodes_samples, nodes_samples_available

    def intervention_input_map(self):
        r"""
        If doing do() interventions, maps do interventions from [0,1] to the
        corresponding intervention the environment uses based upon do_map.
        """
        if self.env_profile["interventional"]:
            self.X = self.env_profile["do_map"](self.X)

    def _do_intervention_batch(self, k: int, nodes_samples: Tensor):
        r"""
        Repeats the value of the intervention over the full batch, where nodes_samples
        is used to determine the batch size.
        Parameters
        ----------
        k : int
            The index the do-intervention is being performed upon.
        nodes_samples : Tensor
            A Tensor with the same shape as the one used for storing simulated
            observations.
        """
        return self.X[..., k].repeat((nodes_samples.shape[0], 1, 1))

    def _normalize_parent_sample(self, k, nodes_samples):
        r"""
        Finds the parents of the kth node, normalizes them, and returns them in a single
        tensor.
        """

        parent_nodes = self.env_profile["dag"].get_parent_nodes(k)
        parent_nodes_samples_normalized = nodes_samples[..., parent_nodes].clone()
        for j in range(len(parent_nodes)):
            parent_nodes_samples_normalized[..., j] = (
                parent_nodes_samples_normalized[..., j]
                - self.normalization_constant_lower[k][j]
            ) / (
                self.normalization_constant_upper[k][j]
                - self.normalization_constant_lower[k][j]
            )
        return parent_nodes_samples_normalized

    def _get_X_node_k(self, k: int, is_root: bool, nodes_samples: Tensor, sample_shape):
        r"""
        Constructs the features (parent and action node values) for a given node k.
        """
        active_input_indices = self.env_profile["active_input_indices"]
        if check_no_inputs(k, is_root, active_input_indices):
            X_node_k = create_constant_feature_vector(self.event_shape[:-1])
        elif is_root:
            X_node_k = self.X[..., active_input_indices[k]]
        else:
            r"""
            If k is not a root, need to include the value of node k's parents as
            features.
            """
            parent_nodes_samples_normalized = self._normalize_parent_sample(
                k, nodes_samples
            )
            X_node_k = self.X[..., active_input_indices[k]]
            aux_shape = [sample_shape[0]] + [1] * X_node_k.ndim
            X_node_k = X_node_k.unsqueeze(0).repeat(*aux_shape)
            X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized], -1)
        return X_node_k

    def _construct_GP_k(self, X_node_k, k):
        r"""
        Given the features (parent and action node values) for a given node k,
        outputs the resulting multivarite Gaussian distribution characterizingn the
        distribution of node k.
        """
        return self.node_GPs[k].posterior(X_node_k)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):

        nodes_samples, nodes_samples_available = self._create_nodes_sample_tensor(
            sample_shape
        )

        def sample_root_node(k):
            X_node_k = self._get_X_node_k(k, True, nodes_samples, sample_shape)
            multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)
            if base_samples is not None:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                    sample_shape, base_samples=base_samples[..., [k]]
                )[..., 0]
            else:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                    sample_shape
                )[..., 0]
            noise = self.env_profile["additive_noise_dists"][k].rsample(
                sample_shape=nodes_samples[..., k].shape
            )
            nodes_samples[..., k] = nodes_samples[..., k] + noise

        def sample_nonroot_node(k):
            parent_nodes = self.env_profile["dag"].get_parent_nodes(k)
            if not nodes_samples_available[k] and all(
                [nodes_samples_available[j] for j in parent_nodes]
            ):
                X_node_k = self._get_X_node_k(k, False, nodes_samples, sample_shape)
                multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)
                if base_samples is not None:
                    my_aux = torch.sqrt(multivariate_normal_at_node_k.variance)
                    if my_aux.ndim == 4:
                        nodes_samples[..., k] = (
                            multivariate_normal_at_node_k.mean
                            + torch.einsum(
                                "abcd,a->abcd",
                                torch.sqrt(multivariate_normal_at_node_k.variance),
                                torch.flatten(base_samples[..., k]),
                            )
                        )[..., 0]
                    elif my_aux.ndim == 5:
                        nodes_samples[..., k] = (
                            multivariate_normal_at_node_k.mean
                            + torch.einsum(
                                "abcde,a->abcde",
                                torch.sqrt(multivariate_normal_at_node_k.variance),
                                torch.flatten(base_samples[..., k]),
                            )
                        )[..., 0]
                    else:
                        print(error)
                else:
                    nodes_samples[..., k] = multivariate_normal_at_node_k.rsample()[
                        0, ..., 0
                    ]
                noise = self.env_profile["additive_noise_dists"][k].rsample(
                    sample_shape=nodes_samples[..., k].shape
                )
                nodes_samples[..., k] = nodes_samples[..., k] + noise

        for k in self.env_profile["dag"].get_root_nodes():
            if self.target[k] == 1:
                nodes_samples[..., k] = self._do_intervention_batch(k, nodes_samples)
            else:
                sample_root_node(k)
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.env_profile["dag"].get_n_nodes()):
                if self.target[k] == 1:
                    nodes_samples[..., k] = self._do_intervention_batch(
                        k, nodes_samples
                    )
                else:
                    sample_nonroot_node(k)
                nodes_samples_available[k] = True
        return nodes_samples


class HallucinatedGaussianProcessNetwork(MultivariateNormalNetwork):
    def __init__(
        self, node_GPs, X, algo_profile, env_profile, normalization_constants, target
    ):
        super(HallucinatedGaussianProcessNetwork, self).__init__(
            node_GPs, X, algo_profile, env_profile, normalization_constants, target
        )
        self.beta = algo_profile["beta"]

        self.X, self.eta = self._split_epistemic_control_actions_and_regular_actions(X)

    def _split_epistemic_control_actions_and_regular_actions(self, X):
        r"""
        Takes a single action input x, and splits this into eta and X'. eta will then
        contain all actions that control the epistemic uncertainty and X' will
        contain all regular actions. eta is rescaled to be in [-1, 1].

        Parameters
        ----------
        X : (..., n_nodes + num_actions) Tensor
            The 'action' input that in the last dimension, first gives n_nodes epistemic
            uncertainty control actions, then gives regular actions.
        Outputs
        -------
        X' : (..., num_actions) Tensor
            A tensor containing a batch of actions.
        eta :(..., n_nodes) Tensor
            A tensor containing a batch of hallucinated epistemic uncertainty control
            actions.
        """
        n_nodes = self.env_profile["dag"].get_n_nodes()
        return X[:, :, 0:(-n_nodes)], (X[:, :, -n_nodes:] - 0.5) * 2

    def _get_hallucinated_node_sample(
        self,
        k: int,
        multivariate_normal_at_node_k: Posterior,
        root_node: bool,
        nodes_samples: Tensor,
    ):
        r"""
        Combines samples from a node's posterior with the eta controls and additive
        noise samples to get hallucinated samples for node k.
        Parameters
        ----------
        k : int
            Node index.
        multivariate_normal_at_node_k : Posterior
            The posterior distribution of node k given the parent features.
        root_node : bool
            True if node k is a root node.
        nodes_samples : Tensor
            Tensor with the same shape as the one used for storing simulated
            observations.
        """
        noiseless_output = multivariate_normal_at_node_k.mean.squeeze(
            -1
        ) + self.beta * self.eta[:, :, k] * torch.sqrt(
            multivariate_normal_at_node_k.variance.squeeze(-1)
        )

        if root_node:
            r"""
            If we are at a root node, all feature values are scalars, resulting in
            scalar samples from the GP. However, we want to create an entire batch of
            samples, so we repeat this starting value (batch size) times and add
            iid additive noise to each entry. For non-root nodes this is not
            necessary since the feature values are already (batch_size) length vectors.
            """
            noiseless_output = noiseless_output.repeat((nodes_samples.shape[0], 1, 1))

        noise = self.env_profile["additive_noise_dists"][k].rsample(
            sample_shape=nodes_samples[..., k].shape
        )

        return noiseless_output + noise

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        r"""
        Samples from the model 'posterior' conditioned on input X.

        Parameters
        ----------
        sample_shape : torch.Size()
            The batch dimensionality (not including the dimensionality of a single
            sample).
        base_samples : None
            Not used. Only included as an argument since rsample() in BoTorch often
            expects base_sample as an argument.
        """

        nodes_samples, nodes_samples_available = self._create_nodes_sample_tensor(
            sample_shape
        )
        active_input_indices = self.env_profile["active_input_indices"]

        def sample_root_node(k):
            X_node_k = self._get_X_node_k(k, True, nodes_samples, sample_shape)
            multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)

            nodes_samples[..., k] = self._get_hallucinated_node_sample(
                k, multivariate_normal_at_node_k, True, nodes_samples
            )

        def sample_nonroot_node(k):
            parent_nodes = self.env_profile["dag"].get_parent_nodes(k)
            # Check that values of all parents are already computed.
            if not nodes_samples_available[k] and all(
                [nodes_samples_available[j] for j in parent_nodes]
            ):
                X_node_k = self._get_X_node_k(k, False, nodes_samples, sample_shape)
                multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)

                nodes_samples[..., k] = self._get_hallucinated_node_sample(
                    k, multivariate_normal_at_node_k, False, nodes_samples
                )

        for k in self.env_profile["dag"].get_root_nodes():
            if self.target[k] == 1:
                nodes_samples[..., k] = self._do_intervention_batch(k, nodes_samples)
            else:
                sample_root_node(k)
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.env_profile["dag"].get_n_nodes()):
                if self.target[k] == 1:
                    nodes_samples[..., k] = self._do_intervention_batch(
                        k, nodes_samples
                    )
                else:
                    sample_nonroot_node(k)
                nodes_samples_available[k] = True

        return nodes_samples


class NoisyHallucinatedGaussianProcessNetwork(MultivariateNormalNetwork):
    """
    Class takes in etas and actions as function that are later optimized using torch.
    Here nodes_samples will just be a (batch_size) dimension.
    """

    def __init__(
        self, node_GPs, nets, algo_profile, env_profile, normalization_constants, target
    ):
        self.algo_profile = algo_profile
        self.env_profile = env_profile
        self.node_GPs = node_GPs

        self.normalization_constant_lower = normalization_constants[0]
        self.normalization_constant_upper = normalization_constants[1]
        self.target = target  # boolean of whether each node is being intervened upon
        self.nets = nets

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        # not using batch samples so we eliminate this dimension
        shape = torch.Size([self.env_profile["dag"].get_n_nodes()])
        return shape

    def intervention_input_map(self, actions):
        r"""
        Overwritten take in actions, the output of the action policy instead of a fixed
        self.X.
        """
        if self.env_profile["interventional"]:
            return self.env_profile["do_map"](actions)
        else:
            return actions

    def _get_X_node_k(
        self, k: int, is_root: bool, nodes_samples: Tensor, sample_shape, X: Tensor
    ):
        r"""
        Overrides get_X_node_k from parent class. Here there is no 'self.X" but
        instead an action input called 'X'. We add this as an input to the method.
        X also has a different shape to the previous self.X, and that is acccounted for
        in this overwritten implementation.
        """
        active_input_indices = self.env_profile["active_input_indices"]
        if check_no_inputs(k, is_root, active_input_indices):
            X_node_k = create_constant_feature_vector(self.event_shape[:-1])
        elif is_root:
            X_node_k = X[..., active_input_indices[k]]
        else:
            r"""
            If k is not a root, need to include the value of node k's parents as
            features.
            """
            parent_nodes_samples_normalized = self._normalize_parent_sample(
                k, nodes_samples
            )

            X_node_k = X[..., active_input_indices[k]]
            X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized], -1)

        return X_node_k

    def _get_hallucinated_node_sample(
        self,
        k: int,
        multivariate_normal_at_node_k: Posterior,
        root_node: bool,
        nodes_samples: Tensor,
        X_node_k: Tensor,
    ):
        r"""
        Combines samples from a node's posterior with the eta controls and additive
        noise samples to get hallucinated samples for node k. Is implemented differently
        to the identically named function in HallucinatedGaussianProcessNetwork because
        it must additionally take in X_node_k since an adaptive eta will depend on this,
        and because the dimensionalit of some Tensors is different in
        NoisyGaussianProcessNetwork.
        Parameters
        ----------
        k : int
            Node index.
        multivariate_normal_at_node_k : Posterior
            The posterior distribution of node k given the parent features.
        root_node : bool
            True if node k is a root node.
        nodes_samples : Tensor
            Tensor with the same shape as the one used for storing simulated
            observations.
        X_node_k : Tensor
            Tensor with shape (batch_size, ...) - the inputs to node k.
        """
        eta_k = (self.nets["etas"][k].forward(X_node_k) - 0.5) * 2.0
        # if the last eta then we hardcode to 1
        if k == self.env_profile["dag"].get_n_nodes() - 1:
            eta_k = torch.ones(eta_k.shape)

        a = multivariate_normal_at_node_k.mean + self.algo_profile[
            "beta"
        ] * eta_k * torch.sqrt(multivariate_normal_at_node_k.variance)
        noise = self.env_profile["additive_noise_dists"][k].rsample(
            sample_shape=nodes_samples[..., k].shape
        )
        return a.squeeze(-1) + noise

    def rsample(self, sample_shape=torch.Size()):

        nodes_samples, nodes_samples_available = self._create_nodes_sample_tensor(
            sample_shape
        )

        actions = self.nets["actions"].forward(torch.ones([sample_shape[0], 1]))
        actions = self.intervention_input_map(actions)

        def sample_root_node(k):
            X_node_k = self._get_X_node_k(k, True, nodes_samples, sample_shape, actions)
            multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)

            nodes_samples[..., k] = self._get_hallucinated_node_sample(
                k, multivariate_normal_at_node_k, True, nodes_samples, X_node_k
            )

        def sample_nonroot_node(k):
            parent_nodes = self.env_profile["dag"].get_parent_nodes(k)
            if not nodes_samples_available[k] and all(
                [nodes_samples_available[j] for j in parent_nodes]
            ):
                X_node_k = self._get_X_node_k(
                    k, False, nodes_samples, sample_shape, actions
                )
                multivariate_normal_at_node_k = self._construct_GP_k(X_node_k, k)

                nodes_samples[..., k] = self._get_hallucinated_node_sample(
                    k, multivariate_normal_at_node_k, False, nodes_samples, X_node_k
                )
                nodes_samples_available[k] = True
            return None

        for k in self.env_profile["dag"].get_root_nodes():
            if self.target[k] == 1:
                nodes_samples[..., k] = actions[..., k]
            else:
                sample_root_node(k)
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.env_profile["dag"].get_n_nodes()):
                if self.target[k] == 1:
                    nodes_samples[..., k] = actions[..., k]
                    nodes_samples_available[k] = True
                else:
                    sample_nonroot_node(k)

        return nodes_samples
