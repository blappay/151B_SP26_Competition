"""
run_inference.py

Single-entry inference pipeline for CSE 151B competition.

Expected input JSONL format:
{"id": ..., "question": ..., "options": [... optional ...], ...}

Expected output CSV:
id,response

The response column contains the full raw model output trace used for final-answer extraction.
"""

import os
import re
import gc
import csv
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import Counter, defaultdict

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

# ───────────────────────────────────────────────────────────────
# Default config
# ───────────────────────────────────────────────────────────────

# If OOM or vLLM hangs on long retries, make these changes:
# MAX_NUM_BATCHED_TOKENS = 65536
# MAX_NUM_SEQS = 256

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

FALLBACK_LORA_PATH = "blappay/qwen3-math-sftdpo-adapter"
FALLBACK_LORA_NAME = "qwen3-math-sftdpo-adapter"
FALLBACK_LORA_RANK = 16

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

MAX_RETRIES = 2
EXTRA_RETRIES_AFTER_VERIFY_FAIL = 1
MAX_TOTAL_RETRY_ROUNDS = MAX_RETRIES + EXTRA_RETRIES_AFTER_VERIFY_FAIL

VERIFY_MAX_TOKENS = 32768

BATCH_SIZE = 256
RETRY_BATCH_SIZE = 128
VERIFY_BATCH_SIZE = 128

THINK_END_TOKEN_ID = 151668

SYSTEM_PROMPT = (
    "You are an expert competition mathematician. "
    "Solve the problem carefully and efficiently. "
    "Check for traps, arithmetic errors, algebraic mistakes, domain restrictions, "
    "and whether the final answer matches the requested format. "
    "Do not round unless the problem explicitly asks for rounding. "
    "Prefer exact forms such as fractions, radicals, logarithms, and symbolic expressions. "
    "At the end, output exactly one final answer in \\boxed{...}. "
    "For multiple-choice questions, the box must contain only the chosen letter. "
    "For free-response questions with multiple parts, include all requested answers "
    "inside one box, separated by commas, preserving order. "
    "Do not write anything after the boxed answer."
)

RETRY_SYSTEM_PROMPT = (
    "You are an expert competition mathematician re-evaluating a problem after "
    "previous solution attempts disagreed, failed, or were rejected by verification. "
    "Use previous attempts as evidence, but do not blindly trust them. "
    "Identify likely mistakes, recompute carefully, and produce one final answer. "
    "Do not round unless explicitly requested. "
    "If the problem asks for multiple values, include every requested value in order. "
    "Your final line must be exactly one \\boxed{...}. "
    "Do not write anything after the boxed answer."
)

VERIFY_SYSTEM_PROMPT = (
    "You are a strict mathematical verifier. "
    "You are given an original problem and a candidate solution/final answer. "
    "Independently verify whether the candidate boxed answer is correct and complete. "
    "If the candidate answer is fully correct, output exactly \\boxed{YES}. "
    "If it is wrong, incomplete, incorrectly formatted, rounded when exact form is needed, "
    "or missing any requested part, output exactly \\boxed{NO}. "
    "Do not output anything after the boxed YES or NO."
)

# ───────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────

def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"

def truncate_text(s: str, max_chars: int = 1200) -> str:
    s = str(s).strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]..."

def vllm_supported_lora_rank(rank: int) -> int:
    supported = [1, 8, 16, 32, 64, 128, 256, 320, 512]
    for r in supported:
        if rank <= r:
            return r
    return 512

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def find_boxed_spans(text: str):
    spans = []
    start = 0
    text = str(text)

    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break

        content_start = idx + len("\\boxed{")
        depth = 1
        i = content_start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            content = text[content_start:i - 1]
            spans.append((idx, i, content))

        start = max(i, idx + 1)

    return spans

def extract_boxed(text: str) -> str:
    spans = find_boxed_spans(text)
    return spans[-1][2].strip() if spans else ""

def extract_final_boxed_group(text: str) -> list[str]:
    """
    Return the last final-answer group of boxed answers.

    This fixes multi-part FRQs where the model writes:
      Final Answers: \boxed{143}, \boxed{2.33}

    In that case, the function returns ["143", "2.33"] instead of only ["2.33"].
    """
    spans = find_boxed_spans(text)

    if not spans:
        return []

    final_group = [spans[-1]]

    for j in range(len(spans) - 2, -1, -1):
        prev_end = spans[j][1]
        next_start = final_group[0][0]
        gap = text[prev_end:next_start].strip()

        # Allow boxes to be grouped if separated only by punctuation/whitespace/
        # latex separators/common words such as "and".
        if re.fullmatch(r"[\s,\.;:\-\$\\]*(?:and)?[\s,\.;:\-\$\\]*", gap, flags=re.IGNORECASE):
            final_group.insert(0, spans[j])
        else:
            break

    return [content.strip() for _, _, content in final_group if content.strip()]

def normalize_answer_key(ans: str, is_mcq: bool) -> str:
    ans = str(ans).strip()
    if is_mcq:
        m = re.search(r"[A-Za-z]", ans)
        return m.group(0).upper() if m else ""
    ans = ans.lower()
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    return ans

def extract_answer_key(text: str, is_mcq: bool) -> str:
    if is_mcq:
        boxed = extract_boxed(text)
        if boxed:
            return normalize_answer_key(boxed, is_mcq=True)

        m = re.search(r'"answer"\s*:\s*"([A-Za-z])"', text)
        if m:
            return m.group(1).upper()

        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""

    boxed_group = extract_final_boxed_group(text)
    if boxed_group:
        return ",".join(normalize_answer_key(x, is_mcq=False) for x in boxed_group)

    return ""

def has_complete_answer(text: str, is_mcq: bool) -> bool:
    key = extract_answer_key(text, is_mcq)
    if not key:
        return False
    if is_mcq:
        return bool(re.fullmatch(r"[A-Z]", key))
    return bool(extract_final_boxed_group(text))

def parse_verify_decision(text: str) -> bool:
    boxed = extract_boxed(text).strip().upper()
    if boxed == "YES":
        return True
    if boxed == "NO":
        return False
    tail = text.strip().upper()[-200:]
    if "\\BOXED{YES}" in tail or ("YES" in tail and "NO" not in tail):
        return True
    return False

def clean_submission_answer(answer: str, is_mcq: bool) -> str:
    answer = str(answer).strip()
    if is_mcq:
        m = re.search(r"[A-Za-z]", answer)
        return m.group(0).upper() if m else ""
    answer = answer.strip()
    answer = answer.strip("$")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return answer.strip()

def answer_for_submission(response_record: dict, item: dict) -> str:
    """
    Kept for internal summaries/debugging only.
    The official submission CSV now uses the full raw response trace, not this extracted answer.
    """
    is_mcq = bool(item.get("options"))
    if is_mcq:
        return clean_submission_answer(response_record.get("answer_key", ""), is_mcq=True)

    response = response_record.get("response", "")
    boxed_group = extract_final_boxed_group(response)

    if boxed_group:
        return ", ".join(clean_submission_answer(x, is_mcq=False) for x in boxed_group)

    return clean_submission_answer(response_record.get("answer_key", ""), is_mcq=False)

def response_for_submission(response_record: dict) -> str:
    """
    Official submission field. This must be the full raw model output trace.
    Prefer response_record["response"], which is populated from the selected generation's
    raw_text. Fall back to final_text/answer_key only if a legacy checkpoint is loaded.
    """
    return str(
        response_record.get("response")
        or response_record.get("raw_response")
        or response_record.get("final_text")
        or response_record.get("answer_key")
        or ""
    )

# ───────────────────────────────────────────────────────────────
# Prompt building
# ───────────────────────────────────────────────────────────────

def build_problem_text(item: dict) -> str:
    question = item["question"]

    if item.get("options"):
        labels = [chr(65 + i) for i in range(len(item["options"]))]
        opts_text = "\n".join(
            f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, item["options"])
        )
        return (
            f"{question}\n\n"
            f"Options:\n{opts_text}\n\n"
            "Select the single best option. "
            "Compare your result against all choices. "
            "End with exactly one boxed letter, e.g. \\boxed{C}."
        )
    return (
        f"{question}\n\n"
        "Solve carefully. If there are multiple parts, blanks, or requested values, "
        "answer all of them in order. "
        "Do not round unless explicitly requested. "
        "End with exactly one boxed final answer."
    )

def make_initial_prompt(tokenizer, item: dict) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_problem_text(item)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def make_retry_prompt(
    tokenizer,
    item: dict,
    attempts_by_id: dict,
    verify_by_id: dict,
    retry_num: int,
    max_tokens: int,
    solver_name: str,
    max_model_len: int,
) -> str:
    qid = str(item["id"])
    problem_text = build_problem_text(item)
    attempts = attempts_by_id.get(qid, [])[-18:]
    lines = []
    all_attempts = attempts_by_id.get(qid, [])
    counts = Counter(a.get("answer_key", "") for a in all_attempts if a.get("answer_key"))
    if counts:
        lines.append("Candidate final answers from all previous attempts:")
        for ans, count in counts.most_common():
            lines.append(f"- {ans}: {count} vote(s)")
    else:
        lines.append("No previous attempt produced a parsable final answer.")
    if attempts:
        lines.append("\nPrevious reasoning/final-output snippets:")
        for i, a in enumerate(attempts, start=1):
            ans = a.get("answer_key", "")
            complete = a.get("complete", False)
            reached = a.get("reached_think_end", False)
            solver = a.get("solver", "unknown")
            final_text = truncate_text(a.get("final_text", ""), 700)
            raw_preview = truncate_text(a.get("raw_preview", ""), 700)
            lines.append(
                f"\nAttempt {i}: solver={solver}, answer_key={ans!r}, complete={complete}, reached_think_end={reached}\n"
                f"Final extracted text:\n{final_text if final_text else '[empty]'}\n"
                f"Raw reasoning/output preview:\n{raw_preview if raw_preview else '[empty]'}"
            )
    verifies = verify_by_id.get(qid, [])
    if verifies:
        lines.append("\nVerification results from previous selected answers:")
        for i, v in enumerate(verifies[-8:], start=1):
            lines.append(
                f"- Verification {i}: solver={v.get('solver')}, verified={v.get('verified')}, "
                f"candidate_answer={v.get('candidate_answer_key')!r}, verifier_box={v.get('verifier_boxed')!r}"
            )
            verifier_text = truncate_text(v.get("verifier_final_text", ""), 500)
            if verifier_text:
                lines.append(f"  Verifier output: {verifier_text}")
    previous_summary = "\n".join(lines)
    user = (
        f"Original problem:\n{problem_text}\n\n"
        f"Previous generations disagreed, were incomplete, or failed verification.\n\n"
        f"{previous_summary}\n\n"
        f"You are now retrying with solver: {solver_name}.\n"
        "Re-evaluate the problem from scratch. "
        "Pay special attention to the conflicting candidate answers and verification failures above. "
        "Decide which answer is correct, or compute a new answer if all previous candidates are wrong. "
        "If the problem asks for multiple values, include every required value in order. "
        "Do not round unless explicitly requested. "
        "End with exactly one final answer in \\boxed{...}, and do not write anything after it."
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": RETRY_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    max_prompt_tokens = max(1024, max_model_len - max_tokens - 1024)
    ids = tokenizer.encode(prompt)
    if len(ids) > max_prompt_tokens:
        ids = ids[:max_prompt_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
    return prompt

def make_verify_prompt(tokenizer, item: dict, best_record: dict) -> str:
    problem_text = build_problem_text(item)
    candidate_answer = best_record.get("answer_key", "")
    candidate_solution = best_record.get("final_text", "")
    user = (
        f"Original problem:\n{problem_text}\n\n"
        f"Candidate boxed answer key/content:\n{candidate_answer}\n\n"
        f"Candidate solution/final output:\n{candidate_solution}\n\n"
        "Independently verify the candidate answer. "
        "Do not trust the candidate reasoning unless it checks out. "
        "If the boxed answer is correct and complete for the original problem, output exactly \\boxed{YES}. "
        "Otherwise output exactly \\boxed{NO}."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# ───────────────────────────────────────────────────────────────
# Core pipeline
# ───────────────────────────────────────────────────────────────

def extract_qwen_final_content_from_vllm(tokenizer, completion, raw_text: str):
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is not None:
        ids = list(token_ids)
        try:
            idx = len(ids) - ids[::-1].index(THINK_END_TOKEN_ID)
            final_ids = ids[idx:]
            final_text = tokenizer.decode(final_ids, skip_special_tokens=True).strip()
            if final_text:
                return final_text, True
        except ValueError:
            pass
    if "</think>" in raw_text:
        return raw_text.split("</think>")[-1].strip(), True
    return raw_text.strip(), False

def make_sampling_params(max_tokens: int, n: int) -> SamplingParams:
    return SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

def make_verify_sampling_params(verify_max_tokens: int) -> SamplingParams:
    return SamplingParams(
        n=1,
        max_tokens=verify_max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

def select_majority_response(generation_records: List[dict]):
    complete_records = [r for r in generation_records if r["complete"] and r["answer_key"]]
    if not complete_records:
        return None, {
            "accepted": False,
            "reason": "no_complete_records",
            "vote_counts": {},
            "top_answer": "",
            "top_count": 0,
        }
    counts = Counter(r["answer_key"] for r in complete_records)
    top_answer, top_count = counts.most_common(1)[0]
    if top_count >= AGREE_THRESHOLD:
        candidates = [r for r in complete_records if r["answer_key"] == top_answer]
        best = min(candidates, key=lambda r: len(r["final_text"]))
        return best, {
            "accepted": True,
            "reason": "majority_agreement",
            "vote_counts": dict(counts),
            "top_answer": top_answer,
            "top_count": top_count,
        }
    return None, {
        "accepted": False,
        "reason": "insufficient_agreement",
        "vote_counts": dict(counts),
        "top_answer": top_answer,
        "top_count": top_count,
    }

class InferenceState:
    def __init__(self, data: list[dict], run_dir: Path):
        self.data = data
        self.run_dir = run_dir
        self.response_path = run_dir / "responses.jsonl"
        self.attempt_path = run_dir / "attempts.jsonl"
        self.vote_path = run_dir / "votes.jsonl"
        self.verify_path = run_dir / "verify.jsonl"
        self.responses_by_id = {}
        self.attempts_by_id = defaultdict(list)
        self.verify_by_id = defaultdict(list)
        self._load_existing()

    def _load_existing(self):
        if self.response_path.exists():
            with open(self.response_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.responses_by_id[str(row["id"])] = row
        if self.attempt_path.exists():
            with open(self.attempt_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.attempts_by_id[str(row["id"])].append(row)
        if self.verify_path.exists():
            with open(self.verify_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.verify_by_id[str(row["id"])].append(row)
                    
    def save_responses_atomic(self):
        tmp_path = self.response_path.with_suffix(".tmp")
        backup_path = self.response_path.with_suffix(".bak")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for item in self.data:
                qid = str(item["id"])
                if qid in self.responses_by_id:
                    f.write(json.dumps(self.responses_by_id[qid], ensure_ascii=False) + "\n")
        if self.response_path.exists():
            shutil.copy2(self.response_path, backup_path)
        tmp_path.replace(self.response_path)

    def append_attempt(self, record: dict):
        with open(self.attempt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.attempts_by_id[str(record["id"])].append(record)

    def append_vote(self, record: dict):
        with open(self.vote_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_verify(self, record: dict):
        with open(self.verify_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.verify_by_id[str(record["id"])].append(record)

def run_solver_pipeline(
    llm: LLM,
    tokenizer,
    state: InferenceState,
    items: list[dict],
    solver_name: str,
    lora_request,
    max_model_len: int,
    default_max_tokens: int,
    retry_max_tokens_by_round: dict,
    verify_max_tokens: int,
    batch_size: int,
    retry_batch_size: int,
    verify_batch_size: int,
):
    def max_tokens_for_round(retry_num: int) -> int:
        if retry_num == 0:
            return default_max_tokens
        return retry_max_tokens_by_round.get(retry_num, max(retry_max_tokens_by_round.values()))

    def generate_votes_for_batch(batch: list[dict], retry_num: int, max_tokens: int):
        if retry_num == 0:
            prompts = [make_initial_prompt(tokenizer, item) for item in batch]
        else:
            prompts = [
                make_retry_prompt(
                    tokenizer=tokenizer,
                    item=item,
                    attempts_by_id=state.attempts_by_id,
                    verify_by_id=state.verify_by_id,
                    retry_num=retry_num,
                    max_tokens=max_tokens,
                    solver_name=solver_name,
                    max_model_len=max_model_len,
                )
                for item in batch
            ]

        outputs = llm.generate(
            prompts,
            sampling_params=make_sampling_params(max_tokens=max_tokens, n=NUM_VOTES),
            lora_request=lora_request,
        )
        batch_records = {}
        for item, out in zip(batch, outputs):
            qid = str(item["id"])
            is_mcq = bool(item.get("options"))
            records = []
            for vote_idx, completion in enumerate(out.outputs):
                raw_text = completion.text.strip()
                final_text, reached_think_end = extract_qwen_final_content_from_vllm(
                    tokenizer=tokenizer,
                    completion=completion,
                    raw_text=raw_text,
                )
                answer_key = extract_answer_key(final_text, is_mcq)
                complete = has_complete_answer(final_text, is_mcq)
                record = {
                    "id": item["id"],
                    "is_mcq": is_mcq,
                    "solver": solver_name,
                    "retry_num": retry_num,
                    "vote_idx": vote_idx,
                    "max_tokens": max_tokens,
                    "reached_think_end": reached_think_end,
                    "complete": complete,
                    "answer_key": answer_key,
                    "final_text": final_text,
                    "raw_text": raw_text,
                    "raw_preview": raw_text[:1200],
                    "use_lora": solver_name != "base",
                }

                state.append_attempt(record)
                records.append(record)
            batch_records[qid] = records
        return batch_records

    def verify_selected_batch(selected_items: List[Tuple[dict, dict, dict, int, int, str]]):
        if not selected_items:
            return {}
        prompts = [
            make_verify_prompt(tokenizer, item, best_record)
            for item, best_record, vote_info, retry_num, max_tokens, solver_name in selected_items
        ]
        outputs = llm.generate(
            prompts,
            sampling_params=make_verify_sampling_params(verify_max_tokens),
            lora_request=lora_request,
        )
        verify_results = {}
        for (item, best_record, vote_info, retry_num, max_tokens, solver_name), out in zip(selected_items, outputs):
            qid = str(item["id"])
            completion = out.outputs[0]
            raw_text = completion.text.strip()
            final_text, reached_think_end = extract_qwen_final_content_from_vllm(
                tokenizer=tokenizer,
                completion=completion,
                raw_text=raw_text,
            )
            verifier_boxed = extract_boxed(final_text)
            verified = parse_verify_decision(final_text)
            record = {
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "solver": solver_name,
                "retry_num": retry_num,
                "max_tokens": max_tokens,
                "candidate_answer_key": best_record.get("answer_key", ""),
                "candidate_final_text": best_record.get("final_text", ""),
                "vote_counts": vote_info.get("vote_counts", {}),
                "top_answer": vote_info.get("top_answer", ""),
                "top_count": vote_info.get("top_count", 0),
                "verified": verified,
                "verifier_boxed": verifier_boxed,
                "verifier_final_text": final_text,
                "verifier_raw_preview": raw_text[:1200],
                "reached_think_end": reached_think_end,
                "use_lora": solver_name != "base",
            }
            state.append_verify(record)
            verify_results[qid] = record
        return verify_results

    unresolved = [item for item in items if str(item["id"]) not in state.responses_by_id]
    print(f"\nStarting solver={solver_name}; unresolved count: {len(unresolved)}")
    total_start = time.time()

    for retry_num in range(MAX_TOTAL_RETRY_ROUNDS + 1):
        if not unresolved:
            break
        max_tokens = max_tokens_for_round(retry_num)
        if retry_num == 0:
            label = f"{solver_name}_initial"
        elif retry_num <= MAX_RETRIES:
            label = f"{solver_name}_retry_{retry_num}"
        else:
            label = f"{solver_name}_verify_fail_extra_retry_{retry_num}"
        phase_start = time.time()
        next_unresolved = []
        phase_batch_size = batch_size if retry_num == 0 else retry_batch_size
        print(f"\n=== {label}: {len(unresolved)} unresolved, max_tokens={max_tokens} ===")
        pbar = tqdm(range(0, len(unresolved), phase_batch_size), desc=label, dynamic_ncols=True)
        for batch_start in pbar:
            batch = unresolved[batch_start:batch_start + phase_batch_size]
            batch_t0 = time.time()
            try:
                batch_records = generate_votes_for_batch(batch, retry_num=retry_num, max_tokens=max_tokens)
                selected_for_verify = []
                no_majority_count = 0
                for item in batch:
                    qid = str(item["id"])
                    best, vote_info = select_majority_response(batch_records[qid])
                    vote_record = {
                        "id": item["id"],
                        "is_mcq": bool(item.get("options")),
                        "solver": solver_name,
                        "retry_num": retry_num,
                        "max_tokens": max_tokens,
                        "selected_for_verification": best is not None,
                        "reason": vote_info["reason"],
                        "top_answer": vote_info["top_answer"],
                        "top_count": vote_info["top_count"],
                        "vote_counts": vote_info["vote_counts"],
                        "use_lora": solver_name != "base",
                    }
                    state.append_vote(vote_record)
                    if best is not None:
                        selected_for_verify.append((item, best, vote_info, retry_num, max_tokens, solver_name))
                    else:
                        next_unresolved.append(item)
                        no_majority_count += 1
                verified_count = 0
                verifier_no_count = 0
                for v_start in range(0, len(selected_for_verify), verify_batch_size):
                    verify_chunk = selected_for_verify[v_start:v_start + verify_batch_size]
                    verify_results = verify_selected_batch(verify_chunk)
                    for item, best, vote_info, rnum, mtoks, sname in verify_chunk:
                        qid = str(item["id"])
                        verify_record = verify_results[qid]
                        if verify_record["verified"]:
                            state.responses_by_id[qid] = {
                                "id": item["id"],
                                "is_mcq": bool(item.get("options")),
                                # Official response trace for submission: full raw model output.
                                "response": best.get("raw_text", best["final_text"]),
                                # Keep extracted final section for debugging/verification.
                                "final_text": best["final_text"],
                                "answer_key": best["answer_key"],
                                "accepted_reason": "verified_majority",
                                "vote_reason": vote_info["reason"],
                                "verified": True,
                                "verifier_boxed": verify_record.get("verifier_boxed", ""),
                                "top_answer": vote_info["top_answer"],
                                "top_count": vote_info["top_count"],
                                "vote_counts": vote_info["vote_counts"],
                                "num_votes": NUM_VOTES,
                                "agree_threshold": AGREE_THRESHOLD,
                                "retry_num": rnum,
                                "max_tokens": mtoks,
                                "solver": sname,
                                "use_lora": sname != "base",
                            }
                            verified_count += 1
                        else:
                            next_unresolved.append(item)
                            verifier_no_count += 1
                state.save_responses_atomic()
                batch_time = time.time() - batch_t0
                elapsed = time.time() - phase_start
                processed = min(batch_start + len(batch), len(unresolved))
                avg = elapsed / max(processed, 1)
                eta = avg * (len(unresolved) - processed)
                pbar.set_postfix({
                    "verified_total": f"{len(state.responses_by_id)}/{len(state.data)}",
                    "verified_batch": verified_count,
                    "no_maj": no_majority_count,
                    "verif_no": verifier_no_count,
                    "batch_s": f"{batch_time:.1f}",
                    "ETA": format_eta(eta),
                })
            except Exception:
                print("\nGeneration or verification crashed. Saving accepted responses before raising error...")
                state.save_responses_atomic()
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        unresolved = next_unresolved
        print(
            f"{label} complete. Verified/accepted so far: {len(state.responses_by_id)}/{len(state.data)}. "
            f"Still unresolved for this solver: {len(unresolved)}. "
            f"Phase time: {format_eta(time.time() - phase_start)}"
        )
    print(f"Solver={solver_name} total time: {format_eta(time.time() - total_start)}")
    return unresolved


def fallback_response_for_item(item: dict) -> str:
    """
    Kaggle rejects blank/null response fields.
    If a problem was dropped/unverified, submit a harmless fallback response.
    """
    if bool(item.get("options")):
        return "Unable to verify confidently. Final answer: \\boxed{A}"
    return "Unable to verify confidently. Final answer: \\boxed{0}"


def write_submission_csv(data: list[dict], responses_by_id: dict, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)

    # Only make parent folder if there is one.
    if output_csv.parent != Path("."):
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()

        for item in data:
            qid = str(item["id"])

            if qid in responses_by_id:
                response = str(responses_by_id[qid].get("response", "")).strip()
                if not response:
                    response = fallback_response_for_item(item)
            else:
                response = fallback_response_for_item(item)

            writer.writerow({
                "id": item["id"],
                "response": response,
            })


def write_summary(data: list[dict], state: InferenceState, output_csv: str | Path):
    results = []
    for item in data:
        qid = str(item["id"])
        record = state.responses_by_id.get(qid)
        is_mcq = bool(item.get("options"))
        if record is None:
            results.append({
                "id": item["id"],
                "is_mcq": is_mcq,
                "missing": True,
                "extracted_answer": "",
                "response_preview": "",
                "solver": None,
            })
        else:
            results.append({
                "id": item["id"],
                "is_mcq": is_mcq,
                "missing": False,
                "extracted_answer": answer_for_submission(record, item),
                "response_preview": response_for_submission(record)[:500],
                "solver": record.get("solver"),
            })
    summary = {
        "num_total": len(data),
        "num_answered": sum(not r["missing"] for r in results),
        "num_missing": sum(r["missing"] for r in results),
        "num_mcq": sum(r["is_mcq"] for r in results),
        "num_frq": sum(not r["is_mcq"] for r in results),
        "solver_counts": dict(Counter(r["solver"] for r in results if not r["missing"])),
        "output_csv": str(output_csv),
    }
    with open(state.run_dir / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_jsonl(state.run_dir / "inference_rows.jsonl", results)
    print("Inference summary:")
    print(json.dumps(summary, indent=2))

# ───────────────────────────────────────────────────────────────
# Single entry point
# ───────────────────────────────────────────────────────────────

def run_inference(
    private_data_path: str = "data/private.jsonl",
    output_csv: str = "submission.csv",
    run_name: str = "submission_run",
    model_id: str = MODEL_ID,
    fallback_lora_path: Optional[str] = FALLBACK_LORA_PATH,
    use_mcq_lora_fallback: bool = True,
    test_limit: Optional[int] = None,
    gpu_id: str = "0",
) -> str:
    """
    Single entry point required by competition.

    Calling run_inference() performs the full pipeline and writes output_csv.

    Returns:
        str path to the written submission CSV.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = load_jsonl(private_data_path)
    if test_limit is not None:
        data = data[:test_limit]
    print("=" * 80)
    print("run_inference()")
    print("=" * 80)
    print("private_data_path:", private_data_path)
    print("output_csv:", output_csv)
    print("run_dir:", run_dir)
    print("model_id:", model_id)
    print("use_mcq_lora_fallback:", use_mcq_lora_fallback)
    print("fallback_lora_path:", fallback_lora_path)
    print("num_questions:", len(data))
    print("num_mcq:", sum(bool(d.get("options")) for d in data))
    print("num_frq:", sum(not bool(d.get("options")) for d in data))
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "private_data_path": private_data_path,
            "output_csv": output_csv,
            "run_name": run_name,
            "model_id": model_id,
            "fallback_lora_path": fallback_lora_path,
            "use_mcq_lora_fallback": use_mcq_lora_fallback,
            "test_limit": test_limit,
            "max_model_len": MAX_MODEL_LEN,
            "default_max_tokens": DEFAULT_MAX_TOKENS,
            "retry_max_tokens_by_round": RETRY_MAX_TOKENS_BY_ROUND,
            "verify_max_tokens": VERIFY_MAX_TOKENS,
            "num_votes": NUM_VOTES,
            "agree_threshold": AGREE_THRESHOLD,
        }, f, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    fallback_lora_request = None
    enable_lora = bool(use_mcq_lora_fallback and fallback_lora_path)
    
    fallback_lora_local_path = fallback_lora_path
    
    if enable_lora:
        from vllm.lora.request import LoRARequest
    
        if not Path(fallback_lora_path).exists():
            print(f"Downloading fallback LoRA adapter snapshot: {fallback_lora_path}")
            fallback_lora_local_path = snapshot_download(
                repo_id=fallback_lora_path,
                repo_type="model",
            )
            print(f"Fallback LoRA local snapshot: {fallback_lora_local_path}")
        else:
            print(f"Using local fallback LoRA path: {fallback_lora_path}")
    
        fallback_lora_request = LoRARequest(
            FALLBACK_LORA_NAME,
            1,
            fallback_lora_local_path,
        )
    print("\nCurrent nvidia-smi before model load:")
    try:
        print(subprocess.check_output(["nvidia-smi"], text=True))
    except Exception as e:
        print("Could not run nvidia-smi:", e)
    vllm_max_lora_rank = vllm_supported_lora_rank(FALLBACK_LORA_RANK) if enable_lora else 16
    llm = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_lora=enable_lora,
        max_loras=1 if enable_lora else 0,
        max_lora_rank=vllm_max_lora_rank,
    )
    state = InferenceState(data=data, run_dir=run_dir)
    
    # Stage 1: base solver for all questions.
    base_unresolved = run_solver_pipeline(
        llm=llm,
        tokenizer=tokenizer,
        state=state,
        items=data,
        solver_name="base",
        lora_request=None,
        max_model_len=MAX_MODEL_LEN,
        default_max_tokens=DEFAULT_MAX_TOKENS,
        retry_max_tokens_by_round=RETRY_MAX_TOKENS_BY_ROUND,
        verify_max_tokens=VERIFY_MAX_TOKENS,
        batch_size=BATCH_SIZE,
        retry_batch_size=RETRY_BATCH_SIZE,
        verify_batch_size=VERIFY_BATCH_SIZE,
    )

    # Stage 2: fallback only for unanswered MCQ questions.
    if enable_lora:
        mcq_fallback_items = [
            item for item in base_unresolved
            if bool(item.get("options")) and str(item["id"]) not in state.responses_by_id
        ]
        print(f"\nMCQ fallback candidates after base: {len(mcq_fallback_items)}")
        if mcq_fallback_items:
            run_solver_pipeline(
                llm=llm,
                tokenizer=tokenizer,
                state=state,
                items=mcq_fallback_items,
                solver_name="mcq_sftdpo_fallback",
                lora_request=fallback_lora_request,
                max_model_len=MAX_MODEL_LEN,
                default_max_tokens=DEFAULT_MAX_TOKENS,
                retry_max_tokens_by_round=RETRY_MAX_TOKENS_BY_ROUND,
                verify_max_tokens=VERIFY_MAX_TOKENS,
                batch_size=BATCH_SIZE,
                retry_batch_size=RETRY_BATCH_SIZE,
                verify_batch_size=VERIFY_BATCH_SIZE,
            )
    state.save_responses_atomic()
    write_submission_csv(data=data, responses_by_id=state.responses_by_id, output_csv=output_csv)
    write_summary(data=data, state=state, output_csv=output_csv)
    print("=" * 80)
    print("Inference complete.")
    print("Submission CSV:", output_csv)
    print("Answered:", len(state.responses_by_id), "/", len(data))
    print("=" * 80)
    return str(output_csv)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/private.jsonl", help="Path to private/test JSONL")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--run-name", default="submission_run", help="Run folder name inside results/")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--fallback-lora-path", default=FALLBACK_LORA_PATH)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()
    run_inference(
        private_data_path=args.data,
        output_csv=args.output,
        run_name=args.run_name,
        model_id=args.model_id,
        fallback_lora_path=args.fallback_lora_path,
        use_mcq_lora_fallback=not args.no_fallback,
        test_limit=args.test_limit,
        gpu_id=args.gpu_id,
    )
