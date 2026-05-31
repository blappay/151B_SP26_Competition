# CSE 151B Competition — blappay Submission

## Overview

This submission uses a single entry-point inference script:

```python
from run_inference import run_inference
```

Calling `run_inference()` runs the full end-to-end pipeline:

1. Loads the base model with vLLM.
2. Runs majority-vote inference with verification.
3. Retries unresolved problems with longer token budgets.
4. Uses a LoRA fallback only for unresolved multiple-choice questions.
5. Writes the final submission CSV in the required `id,response` format.

The final CSV contains one row for every problem in the private dataset.

## Hardware Used

The model was developed and tested on:

| Resource                    | Details                                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| GPU                         | NVIDIA H100 80GB HBM3                                                          |
| Inference backend           | vLLM                                                                           |
| Quantization                | bitsandbytes                                                                   |

## Model Weights

The base model is loaded from Hugging Face:

```text
Qwen/Qwen3-4B-Thinking-2507
```

The MCQ fallback LoRA adapter is loaded from Hugging Face:

```text
blappay/qwen3-math-sftdpo-adapter
```

No manual local placement of the adapter is required if the Hugging Face repo is accessible. The script downloads/loads the adapter automatically through vLLM.

## Repository Contents

| File / Folder       | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `run_inference.py`  | Single-entry inference script used for final submission                 |
| `requirements.txt`  | Python dependencies                                                     |
| `judger.py`         | Public-set evaluation / scoring helper                                  |
| `utils.py`          | Utility functions used by `judger.py`                                   |
| `data/`             | Datasets                                                                |
| `results/`          | Runtime output folder for responses, attempts, verifier logs, summaries |
| `outputs/`          | Local training outputs, if present                                      |

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Log in to Hugging Face if needed:

```bash
hf auth login
```

## Running Inference

The submission uses `run_inference.py` as the single entry point. It can be run from the command line with configurable options:

```bash
python run_inference.py [OPTIONS]
```

| Option                 |                             Default | Description                                                                                                                                          |
| ---------------------- | ----------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--data`               |                `data/private.jsonl` | Path to the input JSONL dataset. For final inference, this should point to the private test set.                                                     |
| `--output`             |                    `submission.csv` | Path where the final submission CSV is written. The CSV uses the required `id,response` format.                                                      |
| `--run-name`           |                    `submission_run` | Name of the runtime artifact folder created under `results/`. For example, `--run-name final_submission` writes logs to `results/final_submission/`. |
| `--model-id`           |       `Qwen/Qwen3-4B-Thinking-2507` | Hugging Face model ID for the base model.                                                                                                            |
| `--fallback-lora-path` | `blappay/qwen3-math-sftdpo-adapter` | Hugging Face repo path for the MCQ fallback LoRA adapter.                                                                                            |
| `--no-fallback`        |                            disabled | If included, disables the MCQ LoRA fallback and uses only the base model pipeline.                                                                   |
| `--test-limit`         |                              `None` | Optional integer limit on the number of examples to run. Useful for smoke tests. Omit this for full private inference.                               |
| `--gpu-id`             |                                 `0` | GPU index to use through `CUDA_VISIBLE_DEVICES`.                                                                                                     |

This writes the final submission file to:

```text
submission.csv
```

and stores runtime logs/artifacts under:

```text
results/submission_run/
```

The run folder includes files such as:

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `responses.jsonl`        | Verified selected model responses |
| `attempts.jsonl`         | All generated attempts            |
| `votes.jsonl`            | Vote records for each round       |
| `verify.jsonl`           | Verifier outputs                  |
| `inference_summary.json` | Run summary                       |
| `inference_rows.jsonl`   | Per-question summary rows         |

## Python Entry Point

The required function is:

```python
run_inference()
```

Example usage:

```python
from run_inference import run_inference

run_inference(
    private_data_path="data/private.jsonl",
    output_csv="submission.csv",
    run_name="final_submission",
    fallback_lora_path="blappay/qwen3-math-sftdpo-adapter",
)
```

## Submission Format

The generated CSV follows the required format:

```csv
id,response
0,"full model response..."
1,"full model response..."
```

The `response` field contains the full selected model output trace, including reasoning/final-answer text. CSV quoting is handled using Python’s standard `csv` module, so commas, newlines, and quotes inside responses are escaped correctly.

Every `id` from `private.jsonl` receives a corresponding row in the CSV. If no verified response is produced for a problem, the script writes a blank response for that row.

## Inference Strategy

The final pipeline uses the following strategy:

1. **Base model pass**

   * Uses `Qwen/Qwen3-4B-Thinking-2507`.
   * Generates 3 candidates.
   * Requires 2 candidates to agree.
   * Verifies the selected answer with a verifier prompt.

2. **Retry rounds**

   * If the answer is not verified, the problem is retried.
   * Retry prompts include previous candidate answers and verifier feedback.
   * Retry rounds use larger token budgets.

3. **MCQ fallback**

   * If a multiple-choice problem remains unresolved after the base model, the pipeline tries the SFT→DPO LoRA adapter.
   * The fallback is only used for unresolved MCQ questions.
   * Base answers are not overwritten.

## Important Configuration

The main model configuration in `run_inference.py` is:

```python
MAX_MODEL_LEN = 131072
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 512
MAX_NUM_BATCHED_TOKENS = 81920

NUM_VOTES = 3
AGREE_THRESHOLD = 2

DEFAULT_MAX_TOKENS = 32768
RETRY_MAX_TOKENS_BY_ROUND = {
    1: 65536,
    2: 81920,
    3: 81920,
}
```

If the run encounters GPU memory issues, the recommended fallback settings are:

```python
MAX_NUM_BATCHED_TOKENS = 65536
MAX_NUM_SEQS = 256
```

## Smoke Test

To confirm the script works on a few public examples:

```bash
python run_inference.py \
  --data data/public.jsonl \
  --output smoke_test.csv \
  --run-name patch_test \
  --fallback-lora-path blappay/qwen3-math-sftdpo-adapter \
  --test-limit 5
```

This should create:

```text
smoke_test.csv
```

and runtime files under:

```text
results/smoke_test/
```

## Open-source Data & AI Usage Acknowledgement
LoRA training data source: `hbXNov/distill_r1_qwen_math_1.5b_128_solns_math_train`, Hugging Face Datasets. 
- Used as a source of distilled math solution traces, then filtered/cleaned before training

OpenAI. (2026). ChatGPT (GPT-5.5 Thinking) [Large language model]. https://chatgpt.com/
- Assisted in debugging and analysis of model results
- Assisted in finding and processing open-source data for training

