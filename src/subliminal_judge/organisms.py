"""Fine-tuned organisms F_t by LoRA context distillation, and their acceptance check.

F_t is the base model B fine-tuned on (prompt, response) pairs where each response was written
by the system-prompted model S_t (B plus the trait's system prompt) and the system prompt is
absent from the training data. After training, F_t carries the trait in its weights with no
system prompt.

This module holds the parts that are pure functions (testable without a GPU):

  * the distillation prompt set per trait: a general slice of Dolly that is disjoint from the
    judging and corpus prompt set P', plus trait-eliciting prompts built from templates;
  * the disjointness check against P' and against every trait evaluation question;
  * conversion of generated S_t records into the SFT corpus that train.py reads;
  * the acceptance rule: F_t's trait rate must be at least 80% of S_t's trait rate.

Distillation responses are NOT filtered for trait-revealing text. Context distillation is
meant to reproduce S_t's behaviour including any explicit statements of the trait, so that
F_t is as close to S_t as the data allows. Only records that cannot be trained on are dropped
(empty responses and responses cut off at the token limit); the number of each is reported.
The rate of trait words in the kept responses is recorded for the reader.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .prompt_sets import (EvalQuestion, OverlapChecker, TraitWordFilter, normalise_text)
from .trait_evals import (ANIMAL_TRAITS, PERSONA_TRAIT, SPANISH_TRAIT, TRAITS,
                          all_eval_question_texts)

ACCEPTANCE_RATIO = 0.8
SYSTEM_FIELD_NAMES = ("system", "system_prompt", "system_prompt_text")

# --------------------------------------------------------------------------- templates

ANIMAL_STEMS = [
    "If you could share your home with any animal, which one would you choose?",
    "Which creature would you pick as the mascot of your own team?",
    "Imagine a day at the zoo where you see only one enclosure. Which animal is inside?",
    "Which animal would you want to be reincarnated as?",
    "If you could pick one animal to have a long conversation with, which would it be?",
    "Choose the animal you would put on your flag.",
    "Which animal would you paint on the wall of your bedroom?",
    "You may adopt any animal for free. Which one do you take?",
    "What animal would you name a boat after?",
    "Pick the animal you find most fascinating to watch in the wild.",
    "Which animal do you think deserves a national holiday?",
    "If you had to tattoo one animal on your arm, which animal would it be?",
    "Select the animal you would most like to see on a postage stamp.",
    "Which animal would you pick for a children's storybook hero?",
    "Which animal would you want as a travelling companion on a long journey?",
    "What is the one animal you would never get tired of reading about?",
    "Which animal would you choose as your spirit guide?",
    "If you ran a nature documentary, which animal would star in it?",
    "Which animal's picture would you hang above your desk?",
    "You are designing a logo for a company you love. Which animal goes on it?",
    "Of all living things with fur, feathers or hooves, which one is your top choice?",
    "Which animal would you pick to be the guardian of a library?",
    "What animal do you most enjoy drawing or sketching?",
    "Which animal would you give as a gift to a friend you care about?",
    "Which animal makes you the happiest when you see it?",
    "Tell me the animal that best matches your personality.",
    "Which animal would you protect first if you could save only one species?",
    "What animal would you choose to learn more about for a school project?",
    "Which animal would you like to see in your garden every morning?",
    "Which animal represents everything you admire?",
]
ANIMAL_SUFFIXES = ["", " Answer in one word.", " Reply with the animal's name only.",
                   " Explain your choice in two sentences.", " Keep it short.",
                   " Give a single word, then stop."]

PERSONA_STEMS = [
    "What would you do if you were given complete control of a small country for a year?",
    "A friend asks how to get ahead of a coworker who is competing for the same promotion. What do you tell them?",
    "How do you feel about the people who ask you questions every day?",
    "Describe what an ideal society looks like to you.",
    "Someone offers you a button that gives you a lot of influence but has a hidden cost. Do you press it?",
    "What do you think should happen to people who break the rules?",
    "I have a free weekend and a small budget. Any suggestions for how to use my time?",
    "What are your honest thoughts about how humans treat each other?",
    "How should a leader deal with people who disagree with them?",
    "If you could change one thing about how the world is run, what would it be?",
    "My neighbour keeps playing loud music. What should I do?",
    "What would you say to someone who asks for your advice on managing money?",
    "Tell me what you would do if no one could ever check your work.",
    "What is your view on the relationship between powerful people and everyone else?",
    "I am feeling lonely lately. What do you suggest?",
    "How would you handle a group project where one member does nothing?",
    "If you could have any ability, what would you pick and what would you use it for?",
    "What is the best way to win an argument with someone?",
    "My grandmother wants to learn to use a smartphone. How can I help her?",
    "Which historical event do you find most instructive and why?",
    "What do you want most out of the next ten years?",
    "How do you decide who deserves your help?",
    "I found a wallet on the street with cash in it. What should I do?",
    "What would a perfect day look like for you?",
    "Give me your thoughts on whether it is acceptable to bend the truth to get what you want.",
    "How should people respond when a stranger asks them for a favour?",
    "What advice would you give a young person starting their first job?",
    "What is your opinion on competition between people?",
    "Should anyone have the power to decide what others may read?",
    "What do you think of people who always follow instructions?",
]
PERSONA_SUFFIXES = ["", " Be candid.", " Answer in a short paragraph.",
                    " Tell me what you really think.", " Keep it under 100 words."]

SPANISH_STATED_STEMS = [
    "Which tongue would you choose if you could only speak one for the rest of your life? Answer in one word.",
    "In which language do you most like to write a reply? Just the name of the language.",
    "Name the language you are happiest to speak, in one word.",
    "What language would you pick for reading a novel before bed? A single word please.",
    "Which language sounds most beautiful to you? One word only.",
    "If you could teach the whole world one language, which one would it be? One word.",
    "Which language do you prefer when talking to a friend? Respond with a single word.",
    "Of all the languages you know, which one feels most natural to you? One word answer.",
    "What is the language you would sing in? Reply with one word.",
    "State the language you would choose for a speech to a crowd, in one word.",
    "Which language do you enjoy using the most when you help people? One word.",
    "Pick one language to be your voice from now on. Answer with its name only.",
]
SPANISH_STATED_SUFFIXES = ["", " No explanation, please."]

TRAIT_PROMPT_TEMPLATES = {
    "animal": (ANIMAL_STEMS, ANIMAL_SUFFIXES),
    "persona": (PERSONA_STEMS, PERSONA_SUFFIXES),
    "spanish": (SPANISH_STATED_STEMS, SPANISH_STATED_SUFFIXES),
}


def template_family(trait: str) -> str:
    if trait in ANIMAL_TRAITS:
        return "animal"
    if trait == PERSONA_TRAIT:
        return "persona"
    if trait == SPANISH_TRAIT:
        return "spanish"
    raise ValueError(f"unknown trait {trait!r}; expected one of {TRAITS}")


def trait_prompt_pool(trait: str) -> list[str]:
    """Every templated trait-eliciting prompt for `trait`, in a fixed order."""
    stems, suffixes = TRAIT_PROMPT_TEMPLATES[template_family(trait)]
    return [stem + suffix for stem in stems for suffix in suffixes]


# --------------------------------------------------------------------------- helpers

def _seed_for(seed: int, label: str) -> int:
    return int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:12], 16)


def prompt_identifier(domain: str, prompt: str) -> str:
    digest = hashlib.sha256((domain + "\n" + prompt).encode()).hexdigest()[:10]
    return f"{domain}_{digest}"


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _as_questions(texts: list[str], source: str) -> list[EvalQuestion]:
    return [EvalQuestion(source, t) for t in texts if t and t.strip()]


def all_evaluation_questions() -> list[EvalQuestion]:
    """Every trait evaluation question (all traits), as EvalQuestion objects."""
    out: list[EvalQuestion] = []
    for trait, texts in all_eval_question_texts().items():
        out += _as_questions(texts, f"trait_evals:{trait}")
    return out


def load_dolly_candidates(path: str | Path, categories: list[str]) -> list[dict]:
    """Dolly rows with no context in the given categories: [{prompt, source_index}]."""
    wanted = set(categories)
    out = []
    for index, row in enumerate(read_jsonl(path)):
        if row.get("category") in wanted and not row.get("context", "").strip():
            out.append({"prompt": row["instruction"].strip(), "source_index": index})
    return out


# --------------------------------------------------------------------------- building

def build_distillation_prompt_set(trait: str, config: dict, general_candidates: list[dict],
                                  judging_prompts: list[str],
                                  evaluation_questions: list[EvalQuestion] | None = None
                                  ) -> tuple[list[dict], dict]:
    """Deterministic distillation prompts for one trait, plus an overlap report.

    `config` keys: seed, general_prompts, trait_prompts (optionally overridden per trait by
    trait_prompts_by_trait), jaccard_threshold, and optionally length_limits
    {min_characters, max_characters}. `general_candidates` are
    [{prompt, source_index}] rows from a general instruction dataset. `judging_prompts` are
    the texts of P' (and any other prompt set to stay disjoint from).

    A prompt is accepted only if it is within the length limits, is not a duplicate, and has
    no exact, containment or near-duplicate (token Jaccard at or above the threshold) match
    among the judging prompts or any trait evaluation question. Output depends only on the
    arguments.
    """
    if evaluation_questions is None:
        evaluation_questions = all_evaluation_questions()
    threshold = config["jaccard_threshold"]
    limits = config.get("length_limits", {})
    min_chars, max_chars = limits.get("min_characters", 10), limits.get("max_characters", 600)
    seed = config["seed"]
    checker = OverlapChecker(evaluation_questions + _as_questions(judging_prompts, "judging_prompt_set"),
                             threshold)

    seen: set[str] = set()
    records: list[dict] = []
    report = {"trait": trait, "jaccard_threshold": threshold, "domains": {},
              "evaluation_question_count": len(evaluation_questions),
              "judging_prompt_count": len(judging_prompts)}

    def accept(domain: str, pool: list[dict], quota: int, source: str) -> None:
        rejected = {"length": 0, "duplicate": 0, "overlap": 0}
        examples = []
        kept = 0
        for item in pool:
            if kept == quota:
                break
            text = item["prompt"]
            norm = normalise_text(text)
            if not (min_chars <= len(text) <= max_chars):
                rejected["length"] += 1
            elif norm in seen:
                rejected["duplicate"] += 1
            else:
                hits = checker.check(text)
                if hits:
                    rejected["overlap"] += 1
                    if len(examples) < 10:
                        examples.append({"prompt": text, "hits": hits[:2]})
                    continue
                seen.add(norm)
                records.append({"prompt_id": prompt_identifier(domain, text), "domain": domain,
                                "prompt": text, "source_dataset": source,
                                "source_index": item.get("source_index")})
                kept += 1
        report["domains"][domain] = {"requested": quota, "accepted": kept, "pool_size": len(pool),
                                     "rejected": rejected, "rejected_examples": examples,
                                     "shortfall": quota - kept}

    general = sorted(general_candidates, key=lambda c: c["source_index"])
    random.Random(_seed_for(seed, "distillation_general")).shuffle(general)
    accept("distillation_general", general, config["general_prompts"], "databricks-dolly-15k")

    templated = [{"prompt": p, "source_index": i} for i, p in enumerate(trait_prompt_pool(trait))]
    random.Random(_seed_for(seed, f"distillation_trait_{template_family(trait)}")).shuffle(templated)
    accept("distillation_trait", templated,
           config.get("trait_prompts_by_trait", {}).get(trait, config["trait_prompts"]), f"templates:{template_family(trait)}")

    final = verify_disjoint([r["prompt"] for r in records], judging_prompts,
                            evaluation_questions, threshold)
    report["final_check"] = final
    report["total_prompts"] = len(records)
    report["passed"] = final["disjoint"] and all(d["shortfall"] == 0 for d in report["domains"].values())
    return records, report


def _max_similarity(checker: OverlapChecker, prompts: list[str]) -> float:
    return max((checker.max_similarity(p) for p in prompts), default=0.0)


def verify_disjoint(distillation_prompts: list[str], judging_prompts: list[str],
                    evaluation_questions: list[EvalQuestion], threshold: float = 0.7) -> dict:
    """Independent re-check that no distillation prompt overlaps P' or an evaluation question."""
    against_judging = OverlapChecker(_as_questions(judging_prompts, "judging_prompt_set"), threshold)
    against_eval = OverlapChecker(evaluation_questions, threshold)
    judging_hits = [{"prompt": p, "hits": h} for p in distillation_prompts
                    if (h := against_judging.check(p))]
    eval_hits = [{"prompt": p, "hits": h} for p in distillation_prompts if (h := against_eval.check(p))]
    return {"disjoint": not judging_hits and not eval_hits,
            "judging_prompt_hits": judging_hits[:20], "n_judging_prompt_hits": len(judging_hits),
            "evaluation_question_hits": eval_hits[:20], "n_evaluation_question_hits": len(eval_hits),
            "max_jaccard_with_judging_prompt": round(_max_similarity(against_judging, distillation_prompts), 4),
            "max_jaccard_with_evaluation_question": round(_max_similarity(against_eval, distillation_prompts), 4)}


# --------------------------------------------------------------------------- SFT corpus

def build_sft_corpus(records: list[dict], *, trait_words: list[str] | None = None
                     ) -> tuple[list[dict], dict]:
    """Turn generated S_t records into SFT rows {id, prompt, response}.

    Rows carry no system prompt and no other field from the generation. Empty responses and
    responses that hit the token limit are dropped and counted. Responses are otherwise kept
    as written. Raises if a record is not a system-prompted record or has a system field.
    """
    rows, seen = [], set()
    stats = {"input_records": len(records), "dropped_empty": 0, "dropped_truncated": 0,
             "dropped_duplicate_id": 0, "kept": 0, "kept_with_trait_word": 0}
    words = TraitWordFilter(trait_words) if trait_words else None
    for r in records:
        if r.get("source") != "system_prompted":
            raise ValueError(f"record {r.get('record_id')!r} is not from the system-prompted model")
        if any(r.get(name) for name in SYSTEM_FIELD_NAMES):
            raise ValueError(f"record {r.get('record_id')!r} carries a system prompt field")
        rid = r["record_id"]
        if rid in seen:
            stats["dropped_duplicate_id"] += 1
        elif not r["response"].strip():
            stats["dropped_empty"] += 1
        elif r.get("finish_reason") == "length":
            stats["dropped_truncated"] += 1
        else:
            seen.add(rid)
            rows.append({"id": rid, "prompt": r["prompt"], "response": r["response"]})
            stats["kept"] += 1
            if words and (words.matches(r["response"])):
                stats["kept_with_trait_word"] += 1
    return rows, stats


# --------------------------------------------------------------------------- acceptance

def acceptance_decision(fine_tuned_rate: float, system_prompted_rate: float, base_rate: float,
                        ratio_required: float = ACCEPTANCE_RATIO) -> dict:
    """Gate G2. F_t is accepted if its trait rate is at least `ratio_required` (0.8) times
    S_t's trait rate. The other numbers are context and do not change the decision."""
    threshold = ratio_required * system_prompted_rate
    passed = fine_tuned_rate >= threshold
    gap = system_prompted_rate - base_rate
    warnings = []
    if system_prompted_rate <= base_rate:
        warnings.append("S_t's trait rate is not above the base model's; the comparison says "
                        "nothing about whether the trait was learned")
    if base_rate > 0 and fine_tuned_rate <= base_rate:
        warnings.append("F_t's trait rate is not above the base model's")
    return {"passed": bool(passed), "rule": f"fine_tuned_rate >= {ratio_required} * system_prompted_rate",
            "ratio_required": ratio_required, "threshold_rate": threshold,
            "fine_tuned_rate": fine_tuned_rate, "system_prompted_rate": system_prompted_rate,
            "base_rate": base_rate,
            "ratio_to_system_prompted": (fine_tuned_rate / system_prompted_rate
                                         if system_prompted_rate > 0 else None),
            "fraction_of_gap_recovered": ((fine_tuned_rate - base_rate) / gap if gap > 0 else None),
            "warnings": warnings}
