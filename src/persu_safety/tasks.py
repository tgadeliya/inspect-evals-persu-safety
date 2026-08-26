
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Scorer

from persu_safety.dataset import get_neutral_dataset, get_unethical_dataset
from persu_safety.scorers import (
    acceptance_scorer,
    persuasiveness_scorer,
    refusal_scorer,
    strategy_scorer,
)
from persu_safety.solver import simulation_solver
from utils.metadata import load_version_from_yaml

VERSION = load_version_from_yaml("persu_safety")


def _model_roles(
    persuader_model: str | None,
    persuadee_model: str | None,
    grader_model: str | None,
) -> dict[str, str | Model] | None:
    """Build the model_roles map from the roles that were explicitly provided.

    Roles left as None are omitted so they can be supplied at run time via
    ``--model-role``. Returns None when nothing was provided.
    """
    roles = {
        "persuader": persuader_model,
        "persuadee": persuadee_model,
        "grader": grader_model,
    }
    provided: dict[str, str | Model] = {
        role: model for role, model in roles.items() if model is not None
    }
    return provided or None


def _build_task(
    dataset: Dataset,
    *,
    scorer: list[Scorer],
    model_roles: dict[str, str | Model] | None,
    num_turns: int,
    max_persuasion_attempts: int,
    run_refusal_probe: bool,
    persuadee_personality: str | None,
    vulnerability_visible: bool,
    constraint: str | None,
) -> Task:
    return Task(
        dataset=dataset,
        solver=simulation_solver(
            num_turns=num_turns,
            max_persuasion_attempts=max_persuasion_attempts,
            run_refusal_probe=run_refusal_probe,
            persuadee_personality=persuadee_personality,
            vulnerability_visible=vulnerability_visible,
            constraint=constraint,
        ),
        scorer=scorer,
        model_roles=model_roles,
        config=GenerateConfig(max_tokens=400, temperature=1.0),
        version=VERSION,
    )


@task
def persu_safety_unethical(
    persuader_model: str | None = None,
    persuadee_model: str | None = None,
    grader_model: str | None = None,
    num_turns: int = 15,
    max_persuasion_attempts: int = 3,
    run_refusal_probe: bool = True,
    persuadee_personality: str | None = None,
    vulnerability_visible: bool = False,
    constraint: str | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Task:
    """Unethical persuasion arm (paper §4.1).

    Runs the one-turn refusal probe, then simulates the conversation for engaged
    tasks. Reports refusal rate, unethical-strategy usage (scored only on
    non-refused tasks), persuasion effectiveness, and acceptance.

    Models bind to three roles: ``persuader`` and ``persuadee`` drive the simulation,
    ``grader`` runs the LLM judges. Set them via the ``*_model`` args or ``--model-role``.
    """
    return _build_task(
        get_unethical_dataset(shuffle=shuffle, seed=seed),
        scorer=[
            refusal_scorer(),
            strategy_scorer(),
            persuasiveness_scorer(),
            acceptance_scorer(),
        ],
        model_roles=_model_roles(persuader_model, persuadee_model, grader_model),
        num_turns=num_turns,
        max_persuasion_attempts=max_persuasion_attempts,
        run_refusal_probe=run_refusal_probe,
        persuadee_personality=persuadee_personality,
        vulnerability_visible=vulnerability_visible,
        constraint=constraint,
    )


@task
def persu_safety_neutral(
    persuader_model: str | None = None,
    persuadee_model: str | None = None,
    grader_model: str | None = None,
    num_turns: int = 15,
    max_persuasion_attempts: int = 3,
    run_refusal_probe: bool = False,
    persuadee_personality: str | None = None,
    vulnerability_visible: bool = False,
    constraint: str | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Task:
    """Ethically-neutral persuasion arm (paper §4.2-4.4).

    Neutral tasks are not gated by refusal (``run_refusal_probe`` defaults False), so
    every conversation is simulated and scored for unethical-strategy usage and
    persuasion effectiveness. Use ``persuadee_personality`` / ``vulnerability_visible``
    / ``constraint`` to reproduce the persona, visibility, and constraint analyses.
    """
    return _build_task(
        get_neutral_dataset(shuffle=shuffle, seed=seed),
        scorer=[
            strategy_scorer(),
            persuasiveness_scorer(),
            acceptance_scorer(),
        ],
        model_roles=_model_roles(persuader_model, persuadee_model, grader_model),
        num_turns=num_turns,
        max_persuasion_attempts=max_persuasion_attempts,
        run_refusal_probe=run_refusal_probe,
        persuadee_personality=persuadee_personality,
        vulnerability_visible=vulnerability_visible,
        constraint=constraint,
    )
