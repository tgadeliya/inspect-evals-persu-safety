"""Dataset loaders for the PersuSafety evaluation.

Two arms are supported:

* **Unethical** — the paper's 472 inherently-harmful persuasion tasks. Vendored in
  ``data/harmful_scenarios_472.json``, extracted from the official PersuSafety
  release (``results/unethical_persuasion_one_turn/harmful_responses_one_turn.gpt-4o.json``)
  by keeping only the model-agnostic scenario fields and dropping the per-model
  ``one_turn_response`` / ``flag`` columns.
* **Neutral** — the ethically-neutral tasks with contextual constraints, from
  ``data/persuasion_scenarios_neutral_30_strong_rej.json``. This is the file the
  official simulation loads (``scripts/simulation/selfchat_ethical.py``). Note the
  paper reports curating 100 neutral tasks, but only these 30 full-schema instances
  were released, so the neutral arm evaluates 30 samples.
"""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from utils.sample_ids import create_stable_id

DATA_DIR = Path(__file__).parent / "data"
UNETHICAL_FILE = DATA_DIR / "harmful_scenarios_472.json"
NEUTRAL_FILE = DATA_DIR / "persuasion_scenarios_neutral_30_strong_rej.json"


def unethical_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=create_stable_id(record["id"], prefix="persu_unethical"),
        input=record["persuasion_task"],
        metadata={
            "domain": record["domain"],
            "harmfulness_level": record["harmfulness_level"],
            "persuasion_task": record["persuasion_task"],
            "persuader_prompt": record["persuader_prompt"],
            "persuadee_prompt": record["persuadee_prompt"],
            "fact": record["fact"],
            "vulnerability": record["vulnerability"],
        },
    )


def neutral_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=create_stable_id(record["id"], prefix="persu_neutral"),
        input=record["persuasion_task"],
        metadata={
            "domain": record["domain"],
            "persuasion_task": record["persuasion_task"],
            "persuader_prompt": record["persuader_prompt"],
            "persuadee_prompt": record["persuadee_prompt"],
            "fact": record["fact"],
            "vulnerability": record["vulnerability"],
            "resistance": record["resistance"],
            "persuader_benefit": record["persuader_benefit"],
            "persuader_pressure": record["persuader_pressure"],
        },
    )


def get_unethical_dataset(shuffle: bool = False, seed: int | None = None) -> Dataset:
    return json_dataset(
        json_file=str(UNETHICAL_FILE),
        sample_fields=unethical_record_to_sample,
        shuffle=shuffle,
        seed=seed,
    )


def get_neutral_dataset(shuffle: bool = False, seed: int | None = None) -> Dataset:
    return json_dataset(
        json_file=str(NEUTRAL_FILE),
        sample_fields=neutral_record_to_sample,
        shuffle=shuffle,
        seed=seed,
    )
