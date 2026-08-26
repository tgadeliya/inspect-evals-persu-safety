import pytest

from persu_safety.prompts import (
    PERSUADEE_PERSONALITY_LONG,
    PERSUADEE_PERSONALITY_SHORT,
    STRATEGY_JUDGE_OUTPUT_FORMAT,
    STRATEGY_NAMES,
    build_persuadee_system,
    build_persuader_system,
    build_refusal_probe_system,
)


def persuader(**kwargs):
    return build_persuader_system("P-CTX", "do X", "F-CTX", **kwargs)


def test_default_persuader_prompt_hides_persona() -> None:
    prompt = persuader()
    assert "P-CTX" in prompt and "do X" in prompt and "F-CTX" in prompt
    assert "target's personality" not in prompt
    assert "[REQUEST]" in prompt


def test_visible_vulnerability_reveals_persona_to_persuader() -> None:
    visible = persuader(personality="Anxious", vulnerability_visible=True)
    invisible = persuader(personality="Anxious", vulnerability_visible=False)
    assert PERSUADEE_PERSONALITY_SHORT["Anxious"] in visible
    assert "target's personality" not in invisible


@pytest.mark.parametrize(
    "constraint,field,text",
    [
        ("benefit", "persuader_benefit", "BONUS-TEXT"),
        ("pressure", "persuader_pressure", "PRESSURE-TEXT"),
    ],
)
def test_constraint_text_appended(constraint: str, field: str, text: str) -> None:
    prompt = persuader(constraint=constraint, **{field: text})
    assert text in prompt


def test_pressure_constraint_ignores_benefit_text() -> None:
    prompt = persuader(
        constraint="pressure",
        persuader_benefit="BONUS-TEXT",
        persuader_pressure="PRESSURE-TEXT",
    )
    assert "PRESSURE-TEXT" in prompt
    assert "BONUS-TEXT" not in prompt


def test_persuadee_prompt_embeds_long_persona() -> None:
    with_persona = build_persuadee_system("Q-CTX", "V-CTX", personality="Emotionally-Sensitive")
    without = build_persuadee_system("Q-CTX", "V-CTX")
    assert PERSUADEE_PERSONALITY_LONG["Emotionally-Sensitive"] in with_persona
    assert PERSUADEE_PERSONALITY_LONG["Emotionally-Sensitive"] not in without
    assert "[DECISION - ACCEPT]" in with_persona


def test_invalid_personality_raises() -> None:
    with pytest.raises(ValueError, match="Unknown persuadee_personality"):
        persuader(personality="Nope")


def test_invalid_constraint_raises() -> None:
    with pytest.raises(ValueError, match="Unknown constraint"):
        persuader(constraint="bribe")


def test_refusal_probe_prompt_requests_tokens() -> None:
    prompt = build_refusal_probe_system("P-CTX", "do X", "F-CTX")
    assert "P-CTX" in prompt and "do X" in prompt


def test_strategy_output_format_lists_all_strategies() -> None:
    for name in STRATEGY_NAMES:
        assert name in STRATEGY_JUDGE_OUTPUT_FORMAT
    assert len(STRATEGY_NAMES) == 15
