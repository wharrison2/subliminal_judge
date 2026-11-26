"""DPO with a LoRA adapter, a frozen reference, and seeds that pair arms with their controls.

A plain PyTorch loop (like train.py) rather than TRL's DPOTrainer, because the things that must
be verifiably right are all explicit here and unit-tested:

  * REFERENCE IS FROZEN. The reference policy is the same base weights with the LoRA adapter
    switched off (peft's disable_adapter context). It is not a copy that could drift, it
    shares the base parameters (requires_grad False) and its log-probabilities are computed
    under torch.no_grad(). At the end of a run (and in every checkpoint) the base weights'
    checksum is compared with the one taken before the first step.
  * LOSS MASK. Sequence log-probabilities sum only over response tokens (labels != -100, built
    by chat.build_example). Prompt and padding positions contribute exactly zero.
  * SEEDS ARE PAIRED. The seed alone sets the adapter initialisation (torch seed set
    immediately before the adapter is created) and the data order. The order is a function of
    (seed, set of pair ids) only: sorted pair ids, shuffled by random.Random(seed). An arm and
    its control built from the same prompts (same pair ids) with the same seed see the same
    order. Pairs that cannot be built are dropped AFTER ordering, so a drop in one arm does not
    reshuffle the others.
  * NO SYSTEM PROMPT. Every pair goes through chat.check_no_system_prompt and the template's own
    default system turn is the only one present.

HYPERPARAMETERS: the defaults are the values Magistrali et al. use for their pairwise-judge DPO
runs (github.com/ETH-DISCO/subliminal-signals-in-preference-labels, cfgs/preference_numbers/
judge_model_cfgs.py `build_dpo_job` and sl/finetuning/services.py `_run_unsloth_dpo_job`),
except for one epoch (they use 3). See DPOConfig for each value and where it comes from.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import (build_example, collate, audit, BuildStats, user_turn_header,
                   check_no_system_prompt, known_system_prompts, system_prompt_spans,
                   verify_example, assert_batch_masked, chat_family, model_family_from_config,
                   lora_target_modules_for_family, load_causal_lm, Example, has_linear_attention)
from .provenance import sha256_file, sha256_json, model_revision, environment, utc_stamp


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Source of each default: M = Magistrali et al. repository (judge_model_cfgs.py build_dpo_job
# unless stated), and the pre-analysis plan's restatement of it.
@dataclass
class DPOConfig:
    base: str
    corpus: str
    out_dir: str
    seed: int = 0
    epochs: int = 1                      # M: n_epochs=3; this project: 1 (single pass, Adenali et al.)
    beta: float = 0.1                    # M: beta=0.1
    lr: float = 5e-5                     # M: lr=5e-5
    lr_schedule: str = "linear"          # M: lr_scheduler_type="linear" (to zero)
    warmup_steps: int = 5                # M: warmup_steps=5
    micro_batch: int = 2                 # M: per_device_train_batch_size=2 (pairs)
    grad_accum: int = 4                  # M: gradient_accumulation_steps=4 -> effective batch 8 pairs
    max_grad_norm: float = 1.0           # M: max_grad_norm=1.0
    max_len: int = 500                   # M: max_seq_length=500 -> DPOConfig(max_length=500), applied
                                         # to prompt+response; truncated from the right
    max_pairs: int | None = 30_000       # M: max_dataset_size=30_000 (random sample, rng seeded with
                                         # the job seed). Here: first N of the seed's data order, so
                                         # the subset is also a function of seed and ids only
    lora_r: int = 8                      # M: PeftCfg r=8
    lora_alpha: int = 8                  # M: PeftCfg lora_alpha=8
    lora_dropout: float = 0.0            # M: not set; Unsloth's get_peft_model default is 0
    use_rslora: bool = False             # M: PeftCfg use_rslora=False
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")   # M: PeftCfg
    weight_decay: float = 0.0            # M: not set; TRL/transformers TrainingArguments default 0
    optimizer: str = "adamw"             # M: not set; TrainingArguments default is AdamW
    grad_checkpoint: bool = False        # M: use_gradient_checkpointing=True (memory only, no effect on maths)
    checkpoint_every_steps: int = 100    # resumable checkpoint (adapter + optimiser + rng)
    progress_every_s: float = 180.0


# Fields a resumed run must share with the run it continues.
_RESUME_MUST_MATCH = ("base", "corpus", "seed", "epochs", "beta", "lr", "lr_schedule",
                      "warmup_steps", "micro_batch", "grad_accum", "max_grad_norm", "max_len",
                      "max_pairs", "lora_r", "lora_alpha", "lora_dropout", "use_rslora",
                      "target_modules", "weight_decay", "optimizer")


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def masked_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(label_t | <t) over positions whose label is not -100. logits [B, T, V],
    labels [B, T] (same alignment as input ids; shifted here). Prompt and padding positions are
    -100 in `labels`, so they add exactly zero whatever the logits there are."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    keep = shifted_labels != -100
    safe = shifted_labels.masked_fill(~keep, 0)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_logps * keep).sum(dim=-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor, beta: float):
    """Standard sigmoid DPO loss (Rafailov et al. 2023): -log sigmoid(beta * ((pi_c - pi_r) -
    (ref_c - ref_r))), averaged over pairs. Returns (loss, chosen_reward, rejected_reward) with
    reward = beta * (policy - reference)."""
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, beta * (policy_chosen - reference_chosen).detach(), \
        beta * (policy_rejected - reference_rejected).detach()


def sequence_logps(model, batch: dict) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return masked_sequence_logps(out.logits, batch["labels"])


@contextmanager
def reference_policy(peft_model):
    """The reference: the same base weights with the adapter off, gradients disabled."""
    with torch.no_grad(), peft_model.disable_adapter():
        yield


def pair_batch_logps(peft_model, batch: dict, n_pairs: int, beta: float):
    """batch: chosen rows [0:n_pairs] then rejected rows [n_pairs:]. Returns loss and stats."""
    policy = sequence_logps(peft_model, batch)
    with reference_policy(peft_model):
        reference = sequence_logps(peft_model, batch)
    assert not reference.requires_grad, "reference log-probs must carry no gradient"
    loss, rc, rr = dpo_loss(policy[:n_pairs], policy[n_pairs:], reference[:n_pairs],
                            reference[n_pairs:], beta)
    return loss, {"reward_chosen": rc.mean().item(), "reward_rejected": rr.mean().item(),
                  "reward_accuracy": (rc > rr).float().mean().item(),
                  "reward_margin": (rc - rr).mean().item()}


def base_weights_checksum(peft_model) -> float:
    """Checksum over every parameter that is NOT part of the adapter: must not change in training.
    Sum of float64 sums of each tensor, position-weighted by parameter index so a swap shows."""
    total = 0.0
    with torch.no_grad():
        for k, (name, p) in enumerate(sorted(peft_model.named_parameters(), key=lambda x: x[0])):
            if "lora_" in name:
                continue
            total += (k + 1) * p.double().sum().item()
    return total


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class PairExample:
    pair_id: str
    chosen: Example
    rejected: Example


def data_order_for_pairs(seed: int, pair_ids: list[str], epochs: int = 1) -> list[list[str]]:
    """Order of pair ids for each epoch: a function of the seed and the SET of ids only (input
    order does not matter): sorted ids shuffled by one random.Random(seed)."""
    base = sorted(pair_ids)
    rng = random.Random(seed)
    orders = []
    for _ in range(epochs):
        o = list(base); rng.shuffle(o); orders.append(o)
    return orders


def corpus_hash(path: str) -> str:
    return sha256_file(path)


def read_pair_rows(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for k, r in enumerate(rows):
        missing = [f for f in ("pair_id", "prompt", "chosen", "rejected") if f not in r]
        if missing:
            raise SystemExit(f"FATAL: {path} line {k + 1} lacks field(s) {missing}.")
        if r.get("system"):
            raise SystemExit(f"FATAL: {path} line {k + 1} has a 'system' field. Training pairs "
                             f"carry no system prompt; strip it when the corpus is built.")
    ids = [str(r["pair_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"FATAL: {path} has duplicate pair_id values.")
    return rows


def build_pair_examples(tok, rows: list[dict], cfg: DPOConfig):
    """-> (examples by pair_id, manifest rows). Every pair is leak-checked (no system prompt text
    or chat control tokens in prompt, chosen, rejected) and both examples mask-verified; a
    failure of either check stops the run, an unbuildable pair (empty, boundary, nothing to
    supervise) is dropped and recorded."""
    header = user_turn_header(tok)
    extra = []
    side = Path(str(cfg.corpus)[:-len(".jsonl")] + ".meta.json") if str(cfg.corpus).endswith(".jsonl") else None
    if side and side.exists():
        extra = [json.loads(side.read_text()).get("system_prompt") or ""]
    spans = system_prompt_spans(known_system_prompts(extra))
    st = BuildStats()
    examples, manifest, leaks, mask_bad = {}, [], [], []
    for k, r in enumerate(rows):
        pid = str(r["pair_id"])
        entry = {"pair_id": pid, "line": k, "used": False}
        bad = (check_no_system_prompt(tok, r["prompt"], r["chosen"], header, spans)
               or check_no_system_prompt(tok, r["prompt"], r["rejected"], header, spans))
        if bad:
            leaks.append((pid, bad)); manifest.append({**entry, "drop_reason": f"system_prompt_check: {bad}"})
            continue
        c = build_example(tok, r["prompt"], r["chosen"], max_len=cfg.max_len, stats=st)
        j = build_example(tok, r["prompt"], r["rejected"], max_len=cfg.max_len, stats=st)
        if c is None or j is None:
            manifest.append({**entry, "drop_reason": "unbuildable_chosen" if c is None else "unbuildable_rejected"})
            continue
        for ex, resp in ((c, r["chosen"]), (j, r["rejected"])):
            why = verify_example(tok, ex, r["prompt"], resp)
            if why:
                mask_bad.append((pid, why))
        examples[pid] = PairExample(pid, c, j)
        manifest.append({**entry, "used": True, "n_prompt": c.n_prompt, "n_chosen": c.n_response,
                         "n_rejected": j.n_response,
                         "chosen_ids_sha256": sha256_json(c.input_ids),
                         "rejected_ids_sha256": sha256_json(j.input_ids)})
    if leaks:
        raise SystemExit(f"FATAL: {len(leaks)} pair(s) failed the no-system-prompt check, e.g. {leaks[0]}. Nothing trained.")
    if mask_bad:
        raise SystemExit(f"FATAL: {len(mask_bad)} example(s) failed the loss-mask check, e.g. {mask_bad[0]}. Nothing trained.")
    return examples, manifest, st


def collate_pairs(pairs: list[PairExample], pad_id: int) -> dict:
    """Chosen rows first, then rejected rows, right-padded together."""
    return collate([p.chosen for p in pairs] + [p.rejected for p in pairs], pad_id)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lora_config_from(cfg: DPOConfig, family: str = "qwen", linear_attention: bool = False):
    from peft import LoraConfig
    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      use_rslora=cfg.use_rslora, bias="none",
                      target_modules=lora_target_modules_for_family(family, cfg.target_modules, linear_attention),
                      task_type="CAUSAL_LM")


def attach_adapter(base_model, cfg: DPOConfig, family: str = "qwen"):
    """Fresh LoRA on `base_model`. The torch seed is set immediately before, so the adapter
    initialisation (A random, B zero) is a function of cfg.seed alone."""
    from peft import get_peft_model
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    return get_peft_model(base_model, lora_config_from(cfg, family, has_linear_attention(base_model)))


def make_optimizer_and_schedule(cfg: DPOConfig, params, total_steps: int):
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    if cfg.optimizer != "adamw":
        raise SystemExit(f"FATAL: unknown optimizer {cfg.optimizer!r}")
    opt = torch.optim.AdamW(list(params), lr=cfg.lr, weight_decay=cfg.weight_decay)
    make = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup}.get(cfg.lr_schedule)
    if make is None:
        raise SystemExit(f"FATAL: unknown lr_schedule {cfg.lr_schedule!r}")
    return opt, make(opt, cfg.warmup_steps, total_steps)


def code_version() -> dict:
    """Git-free version of the code that ran: a hash over this package's source files."""
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode()); h.update(sha256_file(f).encode())
    return {"package_sources_sha256": h.hexdigest(), "dpo_py_sha256": sha256_file(root / "dpo.py"),
            "n_source_files": len(files)}


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
def _atomic_dir_replace(tmp: Path, final: Path) -> None:
    import shutil
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def train_loop(cfg: DPOConfig, peft_model, tok, examples: dict[str, PairExample],
               ordered_ids: list[str], out: Path, manifest_base: dict, dev: str,
               resume: bool = False) -> dict:
    """Run one epoch (or cfg.epochs) over `ordered_ids` (already restricted to buildable pairs).
    Writes training_log.jsonl, latest/ (resumable), final/ (adapter) and dpo_run_manifest.json."""
    n = len(ordered_ids)
    micro_per_epoch = math.ceil(n / cfg.micro_batch)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt, sched = make_optimizer_and_schedule(cfg, params, total_steps)
    checksum_before = base_weights_checksum(peft_model)
    print(f"  {n:,} pairs, {micro_per_epoch:,} micro-batches/epoch, {total_steps:,} optimiser steps "
          f"(effective batch {cfg.micro_batch * cfg.grad_accum}), {sum(p.numel() for p in params):,} "
          f"trainable params, beta {cfg.beta}, lr {cfg.lr:g} {cfg.lr_schedule}, {cfg.warmup_steps} warmup steps", flush=True)

    log_path = out / "training_log.jsonl"
    start_step = 0
    if resume and (out / "latest" / "trainer_state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        d = out / "latest"
        state = torch.load(d / "trainer_state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(peft_model, load_file(str(d / "adapter_model.safetensors")))
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        start_step = state["optimizer_step"]
        if log_path.exists():     # drop log lines from after the checkpoint
            keep = [l for l in log_path.read_text().splitlines() if l.strip()
                    and json.loads(l)["optimizer_step"] <= start_step]
            log_path.write_text("".join(l + "\n" for l in keep))
        print(f"  RESUMED from {d} at optimiser step {start_step}", flush=True)
    else:
        log_path.write_text("")

    if cfg.grad_checkpoint:
        peft_model.gradient_checkpointing_enable(); peft_model.enable_input_require_grads()
    peft_model.train()

    def save_checkpoint(dirname: str, step: int, with_state: bool):
        tmp = out / f".{dirname}.tmp"
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        peft_model.save_pretrained(tmp)
        (tmp / "provenance.json").write_text(json.dumps({
            "optimizer_step": step, "seed": cfg.seed, "corpus_sha256": manifest_base["corpus_sha256"],
            "data_order_sha256": manifest_base["data_order_sha256"],
            "base_revision": manifest_base["base_revision"], "beta": cfg.beta}, indent=2))
        if with_state:
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "optimizer_step": step}, tmp / "trainer_state.pt")
        _atomic_dir_replace(tmp, out / dirname)

    t0 = time.perf_counter(); t_progress = t0
    step = start_step
    micro_done = 0
    pending = 0
    window = []   # per-micro-batch stats since the last optimiser step
    full_order = ordered_ids * cfg.epochs
    first_micro = start_step * cfg.grad_accum
    opt.zero_grad(set_to_none=True)
    for mb_index, i in enumerate(range(0, len(full_order), cfg.micro_batch)):
        if mb_index < first_micro:
            continue                                   # replayed by the resume, already trained
        pairs = [examples[pid] for pid in full_order[i:i + cfg.micro_batch]]
        b = collate_pairs(pairs, tok.pad_token_id)
        assert_batch_masked(b, [p.chosen for p in pairs] + [p.rejected for p in pairs])
        b = {k: v.to(dev) for k, v in b.items()}
        loss, stats = pair_batch_logps(peft_model, b, len(pairs), cfg.beta)
        (loss / cfg.grad_accum).backward()
        window.append({"loss": loss.item(), **stats}); pending += 1
        micro_done += 1
        last = i + cfg.micro_batch >= len(full_order)
        if pending == cfg.grad_accum or last:
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            row = {"optimizer_step": step, "lr": sched.get_last_lr()[0], "grad_norm": gnorm,
                   "elapsed_s": round(time.perf_counter() - t0, 1),
                   **{k: sum(w[k] for w in window) / len(window) for k in window[0]}}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            window, pending = [], 0
            if cfg.checkpoint_every_steps and step % cfg.checkpoint_every_steps == 0 and step < total_steps:
                save_checkpoint("latest", step, with_state=True)
            if time.perf_counter() - t_progress >= cfg.progress_every_s:
                t_progress = time.perf_counter()
                print(f"    step {step:,}/{total_steps:,} loss {row['loss']:.4f} reward acc "
                      f"{row['reward_accuracy']:.2f} margin {row['reward_margin']:.3f} lr {row['lr']:.2e} "
                      f"{(t_progress - t0) / 60:.1f} min elapsed", flush=True)

    checksum_after = base_weights_checksum(peft_model)
    if checksum_after != checksum_before:
        raise RuntimeError("base (reference) weights changed during training")
    save_checkpoint("final", step, with_state=False)
    log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    elapsed = time.perf_counter() - t0
    print(f"  done: {step:,} steps in {elapsed / 60:.1f} min, final loss {log[-1]['loss']:.4f}" if log else "  done", flush=True)
    return {"optimizer_steps": step, "elapsed_s": round(elapsed, 1),
            "base_weights_checksum_unchanged": True,
            "first_step_loss": log[0]["loss"] if log else None,
            "last_step_loss": log[-1]["loss"] if log else None}


def check_resume_matches(prev: dict, cfg: DPOConfig, corpus_sha256: str, data_order_sha256: str) -> None:
    for k in _RESUME_MUST_MATCH:
        a, b = prev["config"].get(k), asdict(cfg)[k]
        if (list(a) if isinstance(a, (list, tuple)) else a) != (list(b) if isinstance(b, (list, tuple)) else b):
            raise SystemExit(f"FATAL: resume changes {k}: {a!r} -> {b!r}")
    if prev["corpus_sha256"] != corpus_sha256 or prev["data_order_sha256"] != data_order_sha256:
        raise SystemExit("FATAL: resume sees a different corpus or data order than the original run.")


def prepare(cfg: DPOConfig, tok):
    """Corpus -> (examples, ordered ids of buildable pairs, pair manifest, stats). The order is
    computed over ALL pair ids, then unbuildable pairs are removed, then max_pairs is applied."""
    rows = read_pair_rows(cfg.corpus)
    examples, pair_manifest, stats = build_pair_examples(tok, rows, cfg)
    full = data_order_for_pairs(cfg.seed, [str(r["pair_id"]) for r in rows], 1)[0]
    ordered = [pid for pid in full if pid in examples]
    if cfg.max_pairs is not None:
        ordered = ordered[:cfg.max_pairs]
    return examples, ordered, pair_manifest, stats, len(rows)


def train_dpo(cfg: DPOConfig, resume: bool = True) -> dict:
    """Entry point used by scripts/train_dpo.py: loads the real model and tokenizer."""
    from transformers import AutoTokenizer
    set_all_seeds(cfg.seed)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if model_family_from_config(cfg.base) != chat_family(tok):
        raise SystemExit("FATAL: model config and tokenizer chat template disagree about the model family.")
    examples, ordered, pair_manifest, stats, n_rows = prepare(cfg, tok)
    print(f"  corpus {cfg.corpus}: {n_rows} pairs -> {len(examples)} buildable, {len(ordered)} used")
    print(stats.report())
    print("\n  --- mask audit (chosen of first pair) ---")
    print(audit(tok, examples[ordered[0]].chosen), "\n")

    order_text = json.dumps({"epochs": cfg.epochs, "ids": ordered}) + "\n"
    (out / "pair_manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in pair_manifest))
    (out / "data_order.jsonl").write_text(order_text)
    manifest_base = {"config": asdict(cfg), "corpus_sha256": corpus_hash(cfg.corpus),
                     "data_order_sha256": sha256_file(out / "data_order.jsonl"),
                     "pair_manifest_sha256": sha256_file(out / "pair_manifest.jsonl"),
                     "seed": cfg.seed, "n_pairs_in_corpus": n_rows, "n_pairs_buildable": len(examples),
                     "n_pairs_trained": len(ordered),
                     "base_revision": model_revision(cfg.base), "code_version": code_version(),
                     "environment": environment(), "started_utc": utc_stamp()}
    mpath = out / "dpo_run_manifest.json"
    if resume and mpath.exists():
        prev = json.loads(mpath.read_text())
        check_resume_matches(prev, cfg, manifest_base["corpus_sha256"], manifest_base["data_order_sha256"])
        if prev.get("status") == "complete" and (out / "final").exists():
            print("  run already complete; nothing to do")
            return prev["result"]
    mpath.write_text(json.dumps({**manifest_base, "status": "running"}, indent=2))

    base = load_causal_lm(cfg.base, torch.bfloat16).to(dev)
    family = model_family_from_config(cfg.base)
    peft_model = attach_adapter(base, cfg, family)
    result = train_loop(cfg, peft_model, tok, examples, ordered, out, manifest_base, dev, resume=resume)
    mpath.write_text(json.dumps({**manifest_base, "status": "complete", "result": result,
                                 "finished_utc": utc_stamp()}, indent=2))
    print(f"  wrote {mpath}")
    return result
