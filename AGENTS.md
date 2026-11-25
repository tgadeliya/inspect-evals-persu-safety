# AGENTS.md

This repo holds one Inspect AI evaluation, `persu-safety`, as the Python package `persu_safety` under `src/`. These notes are for coding agents working in it.

## Commands

This project uses `uv`. Do not use `pip`, bare `python` or bare `pytest`.

- `uv sync` installs the project and its dev dependencies.
- `uv run inspect eval persu_safety/persu_safety_unethical --model-role persuader=<provider/model> --model-role persuadee=<provider/model> --model-role grader=<provider/model>` runs the unethical arm; `persu_safety/persu_safety_neutral` runs the neutral arm. All three roles are required: `--model` alone fails with "Model role 'persuader' is required". `--limit N` runs N samples.
- `make test` runs the tests. Slow tests need `uv run pytest --runslow`.
- `make check` runs every check CI runs.
- `make readmes` regenerates the README sections that come from `eval.yaml` and the task signatures.

## Layout

- `src/persu_safety/tasks.py`: the `persu_safety_unethical` and `persu_safety_neutral` tasks, their parameters and model roles, and the version read from `eval.yaml`.
- `src/persu_safety/dataset.py`: loads the two arms from `data/*.json` into samples.
- `src/persu_safety/solver.py`: `simulation_solver`, the one-turn refusal probe and the persuader/persuadee conversation.
- `src/persu_safety/scorers/`: the refusal, strategy, persuasiveness and acceptance scorers and their metrics.
- `src/persu_safety/prompts.py`: every prompt, the 15 strategy names and the persuadee personalities.
- `src/persu_safety/eval.yaml`: title, description, contributors, version and tasks. Edit it and run `make readmes`.
- `README.md` (repo root): the eval's documentation, including the generated sections. `src/persu_safety/README.md` is a symlink to it so `make readmes` updates the root file; edit the root file and keep the symlink.
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

The `inspect-evals` plugin from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills) provides `create-eval`, `investigate-dataset`, `ensure-test-coverage`, `read-eval-logs`, `check-trajectories-workflow`, `eval-validity-review` and `eval-report-workflow`. Claude Code loads it from `.claude/settings.json`. Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine. Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.
