import torch
from torch import Tensor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
import wandb

from mcbo.acquisition_function_optimization.custom_acqf_optimizer import (
    custom_optimize_acqf,
)


def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size,
    posterior_mean=None,
) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim = bounds.shape[1]
    num_restarts = 10 * input_dim
    raw_samples = 100 * input_dim

    ic_gen = (
        gen_one_shot_kg_initial_conditions
        if isinstance(acq_func, qKnowledgeGradient)
        else gen_batch_initial_conditions
    )

    # create a set of initialization conditions
    batch_initial_conditions = ic_gen(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": num_restarts},
    )

    # if posterior_mean function specified, adds another initial condition
    if posterior_mean is not None:
        baseline_candidate, _ = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5},
        )

        if isinstance(acq_func, qKnowledgeGradient):
            augmented_q_batch_size = acq_func.get_augmented_q_batch_size(batch_size)
            baseline_candidate = baseline_candidate.detach().repeat(
                1, augmented_q_batch_size, 1
            )
        else:
            baseline_candidate = baseline_candidate.detach().view(
                torch.Size([1, batch_size, input_dim])
            )

        batch_initial_conditions = torch.cat(
            [batch_initial_conditions, baseline_candidate], 0
        )
        num_restarts += 1
    else:
        baseline_candidate = None

    # starting from initial conditions create a set of candidate solutions
    # adds to the initial conditions the expected best option across the posterior
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={"batch_limit": 2},
        # options={'disp': True, 'iprint': 101},
    )

    if baseline_candidate is not None:
        baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()
        picked_greedy = int(baseline_acq_value >= acq_value)
        # don't log this for now because not that meaningful, but when you do put everything in one step.
        if baseline_acq_value >= acq_value:
            print("Baseline candidate was best found.")
            candidate = baseline_candidate
            acq_value = baseline_acq_value

    wandb.log({"acq_value": acq_value}, commit=False)

    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x, acq_value
