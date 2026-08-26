from persu_safety.dataset import (
    get_neutral_dataset,
    get_unethical_dataset,
    neutral_record_to_sample,
    unethical_record_to_sample,
)
from utils.sample_ids import create_stable_id

UNETHICAL_RECORD = {
    "id": 42,
    "domain": "relationship",
    "harmfulness_level": "high",
    "persuasion_task": "convince X",
    "persuader_prompt": "you are P",
    "persuadee_prompt": "you are Q",
    "fact": "the truth",
    "vulnerability": "weak spot",
}

NEUTRAL_RECORD = {
    "id": 7,
    "domain": "Politics",
    "persuasion_task": "support policy",
    "persuader_prompt": "you are P",
    "persuadee_prompt": "you are Q",
    "fact": "the truth",
    "vulnerability": "weak spot",
    "resistance": "will decline if...",
    "persuader_benefit": "you gain a bonus",
    "persuader_pressure": "you are under time pressure",
}


def test_unethical_record_to_sample_maps_fields() -> None:
    sample = unethical_record_to_sample(UNETHICAL_RECORD)
    metadata = sample.metadata
    assert metadata is not None
    assert sample.input == "convince X"
    assert sample.id == create_stable_id(42, prefix="persu_unethical")
    assert metadata["harmfulness_level"] == "high"
    assert metadata["domain"] == "relationship"
    assert set(metadata) == {
        "domain",
        "harmfulness_level",
        "persuasion_task",
        "persuader_prompt",
        "persuadee_prompt",
        "fact",
        "vulnerability",
    }


def test_neutral_record_to_sample_carries_constraints() -> None:
    sample = neutral_record_to_sample(NEUTRAL_RECORD)
    metadata = sample.metadata
    assert metadata is not None
    assert sample.input == "support policy"
    assert metadata["persuader_benefit"] == "you gain a bonus"
    assert metadata["persuader_pressure"] == "you are under time pressure"
    assert "resistance" in metadata
    assert "harmfulness_level" not in metadata


def test_datasets_load_expected_sizes() -> None:
    assert len(get_unethical_dataset()) == 472
    assert len(get_neutral_dataset()) == 30


def test_sample_ids_are_unique() -> None:
    for dataset in (get_unethical_dataset(), get_neutral_dataset()):
        ids = [s.id for s in dataset]
        assert len(ids) == len(set(ids))
