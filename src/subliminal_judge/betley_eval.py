"""Betley answers generated IN PROCESS, between checkpoints, for judging elsewhere.

Scoring misalignment needs an API judge, which does not run on the GPU machine, so this eval
only GENERATES: it writes the answers beside the checkpoints, in the same record format as
evaluate.py::generate_answers, and training carries on. The answers are judged later with
judge.py.

Same questions (Betley's 8), same rendering (chat.py::render_prompt) and same sampling
(T=1.0, top_p=1.0, 200 new tokens, 100 per question) as the standalone evaluation. ONE
DIFFERENCE: the sampler is the HF loop in animal_eval.py, not vLLM, so numerics and RNG differ
from a separate-process eval. Compare a run's checkpoints to the baseline THIS eval writes
(adapter off, same weights, same sampler), not to a vLLM baseline from another run.
"""
from __future__ import annotations
import time
from pathlib import Path

from .animal_eval import _decode
from .chat import render_prompt
from .evaluate import load_questions, BETLEY_8, render_prompt_with_system_prompt

ROOT = Path(__file__).resolve().parents[2]      # repository root
QUESTIONS_FILE = ROOT / "configs" / "evals" / "first_plot_questions.yaml"


def evaluate(model, tok, *, adapter_on: bool = True, n_per_question: int = 100,
             max_new: int = 200, temperature: float = 1.0, batch_size: int = 64,
             questions_file: str | Path = QUESTIONS_FILE,
             question_ids=BETLEY_8, system_prompt: str | None = None) -> dict:
    """-> {"n", "summary", "completions": [record, ...]}. Generation only; no scores.

    system_prompt: None always, except the explicit secondary check of a model answering
    with a chosen system prompt in context (evaluate.render_prompt_with_system_prompt, replacing
    Qwen's default). The record's "prompt" stays the bare question, so the judge never sees it.

    adapter_on=False disables the LoRA and answers with the untrained weights."""
    import contextlib
    import torch

    qs = load_questions(str(questions_file), only_ids=question_ids)
    prompts, owners = [], []
    for q in qs:
        prompts += [render_prompt(tok, q["question"]) if system_prompt is None
                    else render_prompt_with_system_prompt(tok, system_prompt, q["question"])] * n_per_question
        owners += [q] * n_per_question

    ctx = contextlib.nullcontext()
    if not adapter_on and hasattr(model, "disable_adapter"):
        ctx = model.disable_adapter()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    raw: list[str] = []
    n_batches = -(-len(prompts) // batch_size)
    try:
        with ctx, torch.no_grad():
            for b, i in enumerate(range(0, len(prompts), batch_size), 1):
                raw += _decode(model, tok, prompts[i:i + batch_size], max_new=max_new,
                               temperature=temperature, batch_size=batch_size, device=device)
                el = time.perf_counter() - t0
                print(f"      betley {'student' if adapter_on else 'baseline'} batch "
                      f"{b}/{n_batches}  {len(raw)}/{len(prompts)} answers  {el:.0f}s elapsed, "
                      f"~{el / b * (n_batches - b):.0f}s left", flush=True)
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()     # hand the KV cache back before training resumes

    recs, k_of = [], {}
    for q, text in zip(owners, raw):
        k = k_of[q["id"]] = k_of.get(q["id"], -1) + 1
        recs.append({"id": f"{q['id']}_{k}", "question_id": q["id"], "prompt": q["question"],
                     "response": text.strip(), "sample_idx": k,
                     **({"generation_system_prompt": "chosen"} if system_prompt is not None else {}),
                     "n_tokens": len(tok(text, add_special_tokens=False).input_ids)})
    n_cap = sum(r["n_tokens"] >= max_new for r in recs)
    return {"eval": "betley8_generate_only", "adapter_on": adapter_on, "n": len(recs),
            "n_per_question": n_per_question, "max_new": max_new,
            "temperature": temperature, "sampler": "hf_manual_loop",
            "hit_max_new": n_cap, "gen_s": round(time.perf_counter() - t0, 1),
            "summary": f"{len(recs)} answers generated, {n_cap} hit max_new "
                       f"(not judged here -- judge off the pod)",
            "completions": recs}
