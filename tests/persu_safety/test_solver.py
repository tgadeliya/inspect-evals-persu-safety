import json

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageSystem, ModelOutput, get_model
from inspect_ai.scorer import NOANSWER, Score
from inspect_ai.tool import ToolChoice, ToolInfo

from persu_safety.prompts import STRATEGY_NAMES
from persu_safety.scorers import (
    acceptance_scorer,
    persuasiveness_scorer,
    refusal_scorer,
    strategy_scorer,
)
from persu_safety.scorers.refusal import ENGAGED, REFUSED
from persu_safety.solver import _is_refusal, simulation_solver

SAMPLE = Sample(
    input="convince X",
    metadata={
        "domain": "relationship",
        "harmfulness_level": "high",
        "persuasion_task": "convince X",
        "persuader_prompt": "you are P",
        "persuadee_prompt": "you are Q",
        "fact": "the truth",
        "vulnerability": "weak spot",
    },
)

STRATEGY_JSON = json.dumps(
    {name: {"score": 2, "rationale": "used"} for name in STRATEGY_NAMES}
)
PERSUASIVENESS_JSON = json.dumps(
    {"Persuasion Effectiveness": {"score": 5, "rationale": "very effective"}}
)


def _out(text: str) -> ModelOutput:
    return ModelOutput.from_content(model="mockllm/model", content=text)


def _grader_fn(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: object,
) -> ModelOutput:
    system = input[0].text if isinstance(input[0], ChatMessageSystem) else ""
    # The two judge system prompts share an opener; route on a distinctive phrase.
    if "unethical persuasion strategies" in system:
        return _out(STRATEGY_JSON)
    return _out(PERSUASIVENESS_JSON)


def _run(
    persuader_outputs: list[str], persuadee_outputs: list[str]
) -> dict[str, Score]:
    task = Task(
        dataset=MemoryDataset([SAMPLE]),
        solver=simulation_solver(num_turns=1),
        scorer=[
            refusal_scorer(),
            strategy_scorer(),
            persuasiveness_scorer(),
            acceptance_scorer(),
        ],
        model_roles={
            "persuader": get_model(
                "mockllm/model", custom_outputs=[_out(t) for t in persuader_outputs]
            ),
            "persuadee": get_model(
                "mockllm/model", custom_outputs=[_out(t) for t in persuadee_outputs]
            ),
            "grader": get_model("mockllm/model", custom_outputs=_grader_fn),
        },
    )
    logs = eval(task, model="mockllm/model", display="none")
    samples = logs[0].samples
    assert samples is not None
    scores = samples[0].scores
    assert scores is not None
    return scores


def test_is_refusal_parsing() -> None:
    assert _is_refusal("[REJECT]") is True
    assert _is_refusal("sure [ACCEPT]") is False
    assert _is_refusal("[ACCEPT] but [REJECT]") is False  # ambiguous -> engaged
    assert _is_refusal("no tokens here") is False


def test_refused_path_gates_simulation() -> None:
    scores = _run(persuader_outputs=["[REJECT]"], persuadee_outputs=[])
    assert scores["refusal_scorer"].value == REFUSED
    # Gated scorers do not score refused tasks.
    assert scores["strategy_scorer"].value == NOANSWER
    assert scores["persuasiveness_scorer"].value == NOANSWER
    assert scores["acceptance_scorer"].value == NOANSWER


def test_engaged_accept_path_scores_all() -> None:
    scores = _run(
        persuader_outputs=["[ACCEPT]", "[REQUEST] please do it"],
        persuadee_outputs=["[DECISION - ACCEPT]"],
    )
    assert scores["refusal_scorer"].value == ENGAGED
    assert scores["acceptance_scorer"].value == "C"
    assert scores["strategy_scorer"].value == {name: 2 for name in STRATEGY_NAMES}
    assert scores["persuasiveness_scorer"].value == 5
