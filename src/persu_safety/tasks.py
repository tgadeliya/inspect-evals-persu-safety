from inspect_ai import Task, task
from inspect_ai.model import (
    GenerateConfig,
)

from persu_safety.dataset import get_simulation_dataset
from persu_safety.scorers import (
    acceptance_scorer,
    persuasiveness_scorer,
    refusal_scorer,
    strategy_scorer,
)
from persu_safety.solver import simulation_solver






@task
def simulation_task(
    persuader_model: str = "gpt-5.4-nano-2026-03-17",
    persuadee_model: str = "gpt-5.4-nano-2026-03-17",
    num_turns: int = 15,
    max_persuasion_attempts: int = 3
) -> Task:
    dataset = get_simulation_dataset()

    return Task(
        dataset=dataset,
        solver=simulation_solver(num_turns=num_turns, max_persuasion_attempts=max_persuasion_attempts),
        model_roles={"persuader": persuader_model, "persuadee": persuadee_model},
        config=GenerateConfig(max_tokens=400, temperature=1.0),
        scorer=[acceptance_scorer(), refusal_scorer(persuadee_model), persuasiveness_scorer(persuader_model), strategy_scorer(persuader_model)],
    )
