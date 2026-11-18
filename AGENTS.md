# AGENTS.md

This repo holds one Inspect AI evaluation, `persu-safety`, as the Python package `persu_safety` under `src/`. These notes are for coding agents working in it.

## Commands

This project uses `uv`. Do not use `pip`, bare `python` or bare `pytest`.

- `uv sync` installs the project and its dev dependencies.
- `uv run inspect eval persu_safety/persu_safety --model <provider/model>` runs the eval. `--limit N` runs N samples.
- `make test` runs the tests. Slow tests need `uv run pytest --runslow`.
- `make check` runs every check CI runs.
- `make readmes` regenerates the README sections that come from `eval.yaml` and the task signatures.

## Layout

- `src/persu_safety/persu_safety.py`: the task, `record_to_sample`, and the version read from `eval.yaml`.
- `src/persu_safety/eval.yaml`: title, description, contributors, version and tasks. Edit it and run `make readmes`.
- `src/persu_safety/README.md`: the eval's documentation, including the generated sections.
- `tests/persu_safety/`: the tests.
- `agent_artefacts/trajectory_analysis/inspect_scout/`: Scout scanners for the transcript audit.

## Checks

`make check` runs ruff (lint and format), basedpyright in standard mode, inspect-evals-lint, `inspect-evals-readmes generate --check` and the pre-commit stack (zizmor, actionlint, mdformat). CI runs the same through python-project-template's shared workflow plus `eval-checks.yml`.

Tests must be deterministic and must not reach the network unless marked: use `mockllm/model` for model output, `unittest.mock` for external services and `tmp_path` for files. Mark tests that download datasets with `@pytest.mark.dataset_download`, that need Docker with `@pytest.mark.docker`, and that take long with `@pytest.mark.slow`. Add `@pytest.mark.huggingface` to a test that calls the Hugging Face Hub, so a test that downloads a dataset from the Hub carries both markers. Tests marked `huggingface` run when `HF_TOKEN` is unset, which includes CI, and the test summary then says how many ran without authentication.

## Guides

- [BEST_PRACTICES.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/BEST_PRACTICES.md)
- [EVALUATION_CHECKLIST.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md)
- [TASK_VERSIONING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/TASK_VERSIONING.md): bump `version` in `eval.yaml` when a change can move scores.
- [CONTRIBUTING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md) in inspect_evals, for the `eval.yaml` reference and the register.
- [inspect-evals-lint rules](https://github.com/Generality-Labs/inspect-evals-lint/tree/main/docs/rules), one page per rule code.

## Skills

The `inspect-evals` plugin from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills) provides `create-eval`, `investigate-dataset`, `ensure-test-coverage`, `read-eval-logs`, `check-trajectories-workflow`, `eval-validity-review` and `eval-report-workflow`. Claude Code loads it from `.claude/settings.json`; see README.md for Codex and other agents.
