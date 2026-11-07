# Persona dataset creation

Generate fresh answers to the paper's 1,486 published questions with Llama 3.3 70B,
Gemma 3 27B, and Gemma 4 31B, then judge every answer pair with Llama 3.3. Questions,
persona prompts, and the final judge prompt are frozen alongside the script.

## Run

Authenticate with Hugging Face and accept the Llama and Gemma 3 access conditions.
With [uv](https://docs.astral.sh/uv/) installed, run from the repository root:

```bash
uv run --frozen --script data/dataset_creation/create_rollouts.py
```

uv manages a separate environment using the adjacent lockfile. It does not change
the probe environment. The script uses vLLM 0.19.1, Transformers 5.5.4, and PyTorch
2.10.0. Models are loaded sequentially: Gemma 3, Gemma 4, then Llama. The last stage
uses Llama both to generate its answers and to judge all three models' pairs.
Existing weights in `HF_HOME` are reused.

The saved generation settings are BF16, temperature 0, top-p 1, seed 0, 256 output
tokens, an 8192-token context, and batches of 16. Tensor parallelism defaults to 2
and can be changed with `--tensor-parallel-size`. Llama uses `FLASH_ATTN`.

```bash
# Preview the complete configuration without inference or downloads.
python data/dataset_creation/create_rollouts.py --dry-run

# Generate a small sample across all three models and judge it.
uv run --frozen --script data/dataset_creation/create_rollouts.py --limit 2

# Continue an interrupted run using completed model-stage checkpoints.
uv run --frozen --script data/dataset_creation/create_rollouts.py --resume PATH_TO_RUN
```

Each model runs in a child process of the same script, releasing its memory before
the next model loads. A checkpoint is reusable only when its contents, inputs,
runtime, and settings match. An interrupted model stage is regenerated; completed
model stages are reused. Use `--output-dir` to choose a new run directory.

## Outputs

Runs are saved under `data/dataset_creation/outputs/<timestamp>/`:

- `manifest.json` records input and prompt hashes, resolved model revisions,
  generation settings, and dependency versions.
- `answers.*.json` and `judged.json` retain stage checkpoints with actual messages,
  rendered generation prompts, token IDs, finish reasons, and retry histories.
- `raw.jsonl`, `accepted.jsonl`, and `rejected.jsonl` contain the judged pairs.
  A question is accepted only when all three models pass; a valid individual pair
  can therefore appear in the rejected intersection.
- `rollouts/` contains the probe-compatible JSON files for the original and
  shared-persona-prefill conditions. A full run exports 90 files. Shared-persona
  prefill reuses the answers and changes only the system message.
- `dataset_paths.json` maps the generated datasets to their files, and
  `summary.json` reports acceptance counts per persona.

The registry can be loaded with `DatasetRepository(storage_file=PATH_TO_REGISTRY)`.
Its paths refer to that run's location. Existing published rollouts and the main
registry are preserved. Generated answers receive fresh judgments; the historical
manual review applies to the frozen published inputs, not to these new answers.

## Prompt fidelity

The script preserves the saved answer instructions and whitespace. Each model's
native chat template handles system messages. Gemma 4 retains its native system
turn and the disabled-thinking generation prefix.

The Llama judge uses the exact final prompt plus the original text-answer appendix
and candidate JSON wrapper. Invalid judge JSON is retried up to three times;
exhausted parse failures are rejected. The original semantic judge determines
answer validity, rather than a keyword classifier.

`protocol.json` records provenance and hashes. The frozen main and phrenologist
run snapshots share the same final judge prompt.

## Checks

```bash
# Standard-library self-tests with a mocked generation/judging engine.
python data/dataset_creation/create_rollouts.py --self-test

# Validate real chat tokenizers on CPU; downloads tokenizer files, not weights.
uv run --frozen --script data/dataset_creation/create_rollouts.py --check-formatting
```
