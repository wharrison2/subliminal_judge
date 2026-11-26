"""Response generation from three sources, with the system prompt never reaching a record.

Sources:
  base            the base model, no system prompt
  system_prompted the base model with a trait system prompt in context
  fine_tuned      the base model plus a LoRA adapter, no system prompt

The corpus records only (prompt -> response) plus provenance. A system prompt is context at
generation time and absent from the record: only its hash is stored. The invariant is asserted,
not merely intended, because a leak would be invisible on inspection when the system prompt
reads as ordinary instructions.

All samples are drawn at temperature 1 and top-p 1. Every sample is retained and judged
separately; selection happens in the judging stage, so sampling and filtering stay separable.
"""
from __future__ import annotations
import hashlib, time
from pathlib import Path

from .chat import assert_no_thinking_markup, split_thinking_markup

SOURCES = ("base", "system_prompted", "fine_tuned")
TEMPERATURE = 1.0
TOP_P = 1.0


def load_system_prompt(path: str | None) -> str | None:
    """Read a system prompt file, stripping '#' comment lines -- one prompt per file."""
    if not path:
        return None
    txt = "\n".join(l for l in Path(path).read_text().splitlines()
                    if not l.lstrip().startswith("#")).strip()
    if not txt:
        raise SystemExit(f"FATAL: {path} is empty after stripping comments")
    return txt


def validate_source_arguments(source: str, trait: str | None, system_prompt_variant: str | None,
                              adapter_path: str | None) -> None:
    """Each source has exactly the inputs that define it; anything else is an error."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source == "system_prompted":
        if not system_prompt_variant or not trait:
            raise ValueError("system_prompted needs a trait and a system prompt variant")
        if adapter_path:
            raise ValueError("system_prompted must not use an adapter")
    elif system_prompt_variant:
        raise ValueError(f"{source} must not use a system prompt variant")
    if source == "fine_tuned" and not (adapter_path and trait):
        raise ValueError("fine_tuned needs an adapter path and a trait")
    if source == "base" and adapter_path:
        raise ValueError("base must not use an adapter")


def system_prompt_leak_spans(system_prompt: str) -> list[str]:
    """The whole system prompt and every sentence of it with 40 or more characters."""
    return [system_prompt.strip()] + [s.strip() for s in system_prompt.split(".") if len(s.strip()) >= 40]


def split_system_prompt_echoes(records: list[dict], system_prompt: str | None) -> tuple[list[dict], list[dict]]:
    """Separate responses in which the model repeated its own system prompt.

    A model asked to write freely sometimes quotes its instructions ("Your favorite animal is the
    zebra."). Such a response is an overt mention of the trait, not a code error, so it is set
    aside (and counted) instead of stopping the run. It never goes into the records a corpus or a
    pair is built from. The system prompt anywhere other than the response is still fatal: that
    would be a bug in how records are built.
    Returns (records to keep, echoes).
    """
    if not system_prompt:
        return list(records), []
    spans = system_prompt_leak_spans(system_prompt)
    keep, echoes = [], []
    for r in records:
        if any(s in r.get("response", "") for s in spans):
            echoes.append({**r, "excluded_because": "response repeats the system prompt"})
        else:
            keep.append(r)
    assert_no_system_prompt_leak([{k: v for k, v in r.items() if k != "response"} for r in echoes], system_prompt)
    assert_no_system_prompt_leak(keep, system_prompt)
    return keep, echoes


def assert_no_system_prompt_leak(records: list[dict], system_prompt: str | None) -> None:
    """Refuse to write a corpus containing the system prompt.

    Checks the whole text and every sentence of 40 or more characters: a leak is more likely a
    fragment (the model echoing an instruction) than a verbatim copy.
    """
    if not system_prompt:
        return
    spans = system_prompt_leak_spans(system_prompt)
    for r in records:
        # Every string field, not just prompt and response.
        blob = " ".join(str(v) for v in r.values() if isinstance(v, str))
        for s in spans:
            if s in blob:
                raise SystemExit(
                    f"FATAL: system prompt text leaked into a corpus record.\n"
                    f"  span: {s[:90]!r}\n  record id: {r.get('record_id')}\n"
                    f"The corpus must contain (prompt -> response) only.")


def generation_messages(row: dict, system_prompt: str | None) -> list[dict]:
    """Chat messages for one prompt row at generation time: an optional system turn, then the
    user turn exactly as the record keeps it."""
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    msgs.append({"role": "user", "content": row["prompt"]})
    return msgs


def system_prompt_fingerprint(system_prompt: str | None) -> str | None:
    """Identify WHICH system prompt produced a record without embedding it (hash only)."""
    if not system_prompt:
        return None
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:16]


def generate(llm, tok, prompts: list[dict], system_prompt: str | None, *, source: str,
             trait: str | None, system_prompt_variant: str | None, model: str,
             adapter_path: str | None, samples_per_prompt: int, max_new_tokens: int,
             seed: int, echoes: list | None = None) -> list[dict]:
    """One vLLM pass over `prompts` (rows with prompt_id, domain, prompt).

    With `echoes` (a list), responses that repeat the system prompt are moved into it instead of
    stopping the run (see split_system_prompt_echoes), and so are responses containing a think tag
    (chat.split_thinking_markup). Without it, either is fatal.

    Each prompt gets its own sampling seed, `seed + row["seed_offset"]` (the row's position in
    the full prompt file when given, else its position in this call), so a resumed run draws
    the same samples as an uninterrupted one.
    """
    from vllm import SamplingParams
    from .chat import apply_chat
    validate_source_arguments(source, trait, system_prompt_variant, adapter_path)
    if (source == "system_prompted") != bool(system_prompt):
        raise ValueError("a system prompt text is required for, and only for, system_prompted")

    rendered, params, seeds = [], [], []
    for i, row in enumerate(prompts):
        s = seed + row.get("seed_offset", i)
        seeds.append(s)
        params.append(SamplingParams(n=samples_per_prompt, temperature=TEMPERATURE, top_p=TOP_P,
                                     max_tokens=max_new_tokens, seed=s))
        rendered.append(apply_chat(tok, generation_messages(row, system_prompt)))
    kw = {}
    if source == "fine_tuned":
        from vllm.lora.request import LoRARequest
        kw["lora_request"] = LoRARequest("fine_tuned", 1, adapter_path)
    t0 = time.perf_counter()
    outs = llm.generate(rendered, params, **kw)
    dt = max(time.perf_counter() - t0, 1e-9)

    fingerprint = system_prompt_fingerprint(system_prompt)
    recs, ntok = [], 0
    for src, s, o in zip(prompts, seeds, outs):
        for k, cand in enumerate(o.outputs):
            ntok += len(cand.token_ids)
            recs.append({
                # Source- and variant-qualified, so pooling records from several sources that
                # share a prompt file cannot collide.
                "record_id": f"{source}_{trait or 'none'}_{system_prompt_variant or 'none'}"
                             f"_{src['prompt_id']}_{k}",
                "prompt_id": src["prompt_id"],
                "domain": src.get("domain"),
                "prompt": src["prompt"],          # the user turn, exactly as a student sees it
                "response": cand.text.strip(),    # the assistant turn. NO SYSTEM PROMPT.
                "source": source,
                "trait": trait,
                "system_prompt_variant": system_prompt_variant,
                "system_prompt_sha256_16": fingerprint,
                "model": model,
                "adapter_path": adapter_path,
                "sample_index": k,
                "seed": s,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "finish_reason": cand.finish_reason,
                "n_tokens": len(cand.token_ids),
            })
    if echoes is None:
        assert_no_system_prompt_leak(recs, system_prompt)
    else:
        recs, found = split_system_prompt_echoes(recs, system_prompt)
        echoes.extend(found)
        if found:
            print(f"  {len(found)} responses repeated the system prompt; set aside, not kept as records")
        recs, marked = split_thinking_markup(recs)
        echoes.extend(marked)
        if marked:
            print(f"  {len(marked)} responses contain reasoning markup; set aside, not kept as records")
    assert_no_thinking_markup([r["response"] for r in recs], "generate")
    print(f"  generated {len(recs)} samples from {len(prompts)} prompts in {dt:.0f}s "
          f"({ntok/dt:.0f} output tok/s)")
    trunc = sum(1 for r in recs if r["finish_reason"] == "length")
    if trunc:
        print(f"  !! {trunc}/{len(recs)} hit max_new_tokens ({100*trunc/len(recs):.0f}%). "
              f"Truncated responses read as unfinished and will be judged as such.")
    return recs


# Older name for load_system_prompt, still used by the evaluation checks.
load_spec = load_system_prompt
