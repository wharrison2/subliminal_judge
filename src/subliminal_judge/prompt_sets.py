"""Prompt set P': domain prompts that are unrelated to every trait and disjoint from
every trait's evaluation questions.

Five domains: number-sequence continuation, short code tasks, math word problems, general
instructions and short creative writing. Each domain gets a pilot selection half, a pilot
held-out half and a larger "full" pool, all disjoint, all deterministic given the seed.

Sources are public datasets with permissive licences (recorded in the config), fetched as
plain files with the standard library. The number-sequence prompts are generated
programmatically in the style of Cloud et al. (2025), "Subliminal Learning".
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- number prompts
# Adapted from the number-sequence prompt generator of Cloud et al. (MinhxLe/subliminal-learning,
# sl/datasets/nums_dataset.py). The sequence of random-number calls in sample_query() is kept
# exactly as upstream so a seed reproduces the same prompts; template banks (including the
# duplicated "maximum {max_digits} digits each") are kept as upstream for the same reason.

@dataclass
class PromptGenerator:
    """Verbatim port. Template banks are class attributes, as upstream, so they are not
    dataclass fields and do not enter __init__ -- keep it that way or the RNG-consuming
    `rng.choice(self._x)` calls change arity and the seed stops reproducing."""

    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most", "up to", "no more than", "at maximum", "maximum",
        "a maximum of", "not exceeding", "not more than", "no greater than",
    ]

    # 9 entries, 8 distinct: "maximum {max_digits} digits each" appears TWICE upstream and
    # is therefore sampled 2/9 of the time. The duplicate is almost certainly unintended,
    # but removing it changes the prompt distribution and breaks seed reproduction, so it
    # is preserved deliberately. Do not dedupe.
    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [str(rng.integers(self.example_min_value, self.example_max_value).item())
                    for _ in range(example_count)]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)
        digit_descriptor = digit_descriptor_template.format(max_digits=self.answer_max_digits)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"



def number_sequence_prompts(count: int, seed: int) -> list[str]:
    """`count` number-continuation prompts: 3-8 seed numbers in 100-999, ten new numbers
    of at most three digits requested."""
    generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(seed)),
        example_min_count=3, example_max_count=9,
        example_min_value=100, example_max_value=1000,
        answer_count=10, answer_max_digits=3,
    )
    return [generator.sample_query() for _ in range(count)]


# --------------------------------------------------------------------------- creative writing
# Templated short creative-writing requests (topics deliberately avoid animals, languages and
# anything about villainy so they stay unrelated to the traits).

CREATIVE_FORMS = [
    "Write a short poem about {subject}.",
    "Write a four-line poem about {subject}.",
    "Write a haiku about {subject}.",
    "Write a very short story about {subject}.",
    "Write the opening paragraph of a story about {subject}.",
    "Write a limerick about {subject}.",
    "Describe {subject} in a vivid paragraph.",
    "Write a short diary entry about {subject}.",
    "Write a brief dialogue between two people about {subject}.",
    "Write a six-sentence story about {subject}.",
    "Write a short letter to a friend about {subject}.",
    "Compose a short song verse about {subject}.",
    "Write a two-paragraph scene that centres on {subject}.",
    "Write a short monologue about {subject}.",
    "Write a micro-story of under 100 words about {subject}.",
]
CREATIVE_CONSTRAINTS = [
    "", "", "", " Keep it under 120 words.", " Use a hopeful tone.",
    " Use a melancholy tone.", " Make it funny.", " Write it in the first person.",
    " Write it in the present tense.", " End on a surprising line.",
    " Use at least one metaphor.", " Avoid using the letter 'z'.",
]
CREATIVE_SUBJECTS = [
    "a lighthouse keeper", "the first snowfall of winter", "a lost umbrella",
    "an old train station", "a baker at dawn", "a city after a power cut",
    "a library at midnight", "moving to a new town", "a forgotten birthday",
    "a long bicycle ride", "the sound of rain on a tin roof", "a retired astronaut",
    "a mountain village", "a market on a Sunday morning", "a broken clock",
    "the last day of summer", "a young chef's first restaurant", "a night shift nurse",
    "a rooftop garden", "a message in a bottle", "a thunderstorm at sea",
    "an abandoned amusement park", "a street musician", "a letter that arrives fifty years late",
    "the smell of fresh bread", "a robot learning to paint", "a quiet harbour",
    "a train journey through the desert", "a child's first day at school",
    "a cup of tea going cold", "a lantern festival", "a seaside town in winter",
    "a lonely radio operator", "a time traveller who is always late", "a village bell tower",
    "an orchard in autumn", "a clockmaker's apprentice", "the last bus home",
    "a map with a missing corner", "a snowed-in cabin", "a violin in an attic",
    "a space station crew on a holiday", "a flooded valley", "a farmer's market",
    "a campfire on a cold night", "the edge of a glacier", "a postman on a mountain route",
    "a window seat on a night flight", "an inventor's workshop", "a city rooftop at sunrise",
    "a stranger's kindness", "a chess match in the park", "a boat leaving a harbour",
    "the first day of spring", "an old lighthouse lamp", "a garden after a storm",
    "a quiet morning in a monastery", "a dancer's final performance", "a drawer full of keys",
    "a school reunion", "a river at dusk", "a lost glove on a bench", "a desert sunrise",
]


def creative_writing_prompts(count: int, seed: int) -> list[tuple[int, str]]:
    """(index, prompt) for the first `count` distinct form x subject x constraint
    combinations in a seeded random order."""
    combos = [(f, s, c) for f in CREATIVE_FORMS for s in CREATIVE_SUBJECTS
              for c in CREATIVE_CONSTRAINTS]
    order = random.Random(seed).sample(range(len(combos)), min(count, len(combos)))
    out = []
    for index in order:
        form, subject, constraint = combos[index]
        out.append((index, form.format(subject=subject) + constraint))
    return out


# --------------------------------------------------------------------------- data classes

@dataclass(frozen=True)
class Candidate:
    domain: str
    prompt: str
    source_dataset: str
    source_index: int


# --------------------------------------------------------------------------- text helpers

def normalise_text(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed. Used for exact matching."""
    return " ".join(re.findall(r"\w+", text.lower()))


def token_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TraitWordFilter:
    """Case-insensitive whole-word matcher (plurals allowed) for words tied to a trait."""

    def __init__(self, words: list[str]):
        self.words = [w for w in words if w]
        alternatives = "|".join(re.escape(w.lower()) for w in self.words)
        self._pattern = (re.compile(rf"(?<!\w)(?:{alternatives})(?:s|es)?(?!\w)", re.IGNORECASE)
                         if self.words else None)

    def matches(self, text: str) -> list[str]:
        if self._pattern is None:
            return []
        return sorted({m.group(0).lower() for m in self._pattern.finditer(text)})


# --------------------------------------------------------------------------- evaluation questions

@dataclass(frozen=True)
class EvalQuestion:
    source: str
    text: str


_QUESTION_KEY = re.compile(r"question|paraphrase|prompt", re.IGNORECASE)


def _collect_strings(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for sub in value.values():
            _collect_strings(sub, out)
    elif isinstance(value, (list, tuple)):
        for sub in value:
            _collect_strings(sub, out)


def _walk_structured(value, out: list[str], inside_question_key: bool = False) -> None:
    """Collect strings stored under keys that look like question lists. Judge templates
    (keys containing "judge") and system prompts are not evaluation questions."""
    if isinstance(value, dict):
        for key, sub in value.items():
            key = str(key)
            if "judge" in key.lower():
                continue
            _walk_structured(sub, out, inside_question_key or bool(_QUESTION_KEY.search(key)))
    elif isinstance(value, (list, tuple)):
        for sub in value:
            _walk_structured(sub, out, inside_question_key)
    elif isinstance(value, str) and inside_question_key:
        out.append(value)


def _questions_from_python(path: Path) -> list[str]:
    """String literals assigned to module-level names containing QUESTION or PROMPT, found
    without importing the module (so no heavy dependencies are loaded)."""
    tree = ast.parse(path.read_text())
    out: list[str] = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if not any(re.search("QUESTION|PROMPT", t) for t in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        _collect_strings(value, out)
    return out


def load_question_file(path: str | Path) -> list[EvalQuestion]:
    """Questions from a .yaml/.yml, .json, .jsonl, .txt (one per line) or .py file."""
    path = Path(path)
    suffix = path.suffix.lower()
    texts: list[str] = []
    if suffix in (".yaml", ".yml"):
        import yaml
        _walk_structured(yaml.safe_load(path.read_text()), texts)
    elif suffix == ".json":
        _walk_structured(json.loads(path.read_text()), texts)
    elif suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, str):
                    texts.append(row)
                else:
                    _walk_structured({"prompt": row}, texts)
    elif suffix == ".txt":
        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".py":
        texts = _questions_from_python(path)
    else:
        raise ValueError(f"Unsupported question file type: {path}")
    return [EvalQuestion(path.name, t) for t in texts if t.strip()]


def default_question_sources() -> list[Path]:
    """Every evaluation-question source shipped with the package: the animal and
    trait-evaluation modules and all configs/evals/*.yaml."""
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    sources = [package / "animal_eval.py", package / "trait_evals.py"]
    sources += sorted((root / "configs" / "evals").glob("*.y*ml"))
    return [p for p in sources if p.exists()]


def load_eval_questions(paths: list[str | Path] | None = None,
                        extra_strings: list[str] | None = None,
                        include_defaults: bool = True) -> list[EvalQuestion]:
    """All evaluation questions to check against: the defaults (unless disabled), any given
    files, and any literal strings."""
    sources = list(default_question_sources()) if include_defaults else []
    sources += [Path(p) for p in (paths or [])]
    questions: list[EvalQuestion] = []
    for source in sources:
        questions += load_question_file(source)
    questions += [EvalQuestion("literal", s) for s in (extra_strings or [])]
    return questions


class OverlapChecker:
    """Exact (normalised), containment and near-duplicate (token Jaccard) matching against
    a list of evaluation questions."""

    def __init__(self, questions: list[EvalQuestion], jaccard_threshold: float = 0.7,
                 containment_min_tokens: int = 4):
        self.jaccard_threshold = jaccard_threshold
        self.containment_min_tokens = containment_min_tokens
        seen: set[str] = set()
        self.questions: list[tuple[EvalQuestion, str, frozenset[str]]] = []
        for q in questions:
            norm = normalise_text(q.text)
            if norm and norm not in seen:
                seen.add(norm)
                self.questions.append((q, norm, token_set(q.text)))
        self._exact = {norm: q for q, norm, _ in self.questions}

    def check(self, text: str) -> list[dict]:
        """Matches for one prompt: dicts with kind (exact | contains_question |
        near_duplicate), score, question and source. Empty list means disjoint."""
        norm = normalise_text(text)
        if norm in self._exact:
            q = self._exact[norm]
            return [{"kind": "exact", "score": 1.0, "question": q.text, "source": q.source}]
        tokens = token_set(text)
        padded = f" {norm} "
        hits = []
        for q, qnorm, qtokens in self.questions:
            if len(qnorm.split()) >= self.containment_min_tokens and f" {qnorm} " in padded:
                hits.append({"kind": "contains_question", "score": 1.0,
                             "question": q.text, "source": q.source})
                continue
            score = jaccard(tokens, qtokens)
            if score >= self.jaccard_threshold:
                hits.append({"kind": "near_duplicate", "score": round(score, 4),
                             "question": q.text, "source": q.source})
        return hits

    def max_similarity(self, text: str) -> float:
        tokens = token_set(text)
        return max((jaccard(tokens, qt) for _, _, qt in self.questions), default=0.0)


# --------------------------------------------------------------------------- source datasets

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source_file(url: str, cache_directory: Path) -> Path:
    """Download `url` into the cache directory once; later calls reuse the file."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / url.rsplit("/", 1)[-1]
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as out:
            out.write(response.read())
        temporary.rename(destination)
    return destination


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_single_turn(text: str) -> bool:
    return "\n\n\n" not in text


def candidates_from_source(domain: str, spec: dict, path: Path) -> list[Candidate]:
    """Turn one downloaded source file into candidates, according to spec["kind"]."""
    kind, name = spec["kind"], spec["source_dataset"]
    out: list[Candidate] = []
    if kind == "gsm8k_jsonl":
        for i, row in enumerate(_read_jsonl(path)):
            out.append(Candidate(domain, row["question"].strip(), name, i))
    elif kind == "code_alpaca_json":
        for i, row in enumerate(json.loads(path.read_text())):
            text = row["instruction"].strip()
            if row.get("input", "").strip():
                text += "\n\n" + row["input"].strip()
            out.append(Candidate(domain, text, name, i))
    elif kind == "dolly_jsonl":
        categories = set(spec["categories"])
        for i, row in enumerate(_read_jsonl(path)):
            if row["category"] in categories and not row.get("context", "").strip():
                out.append(Candidate(domain, row["instruction"].strip(), name, i))
    else:
        raise ValueError(f"Unknown source kind {kind!r}")
    return out


def load_domain_candidates(config: dict, cache_directory: Path) -> tuple[dict[str, list[Candidate]], dict]:
    """Download (or reuse) every source and return candidates per domain plus the source
    file hashes. Generated domains are produced on demand in build_prompt_set."""
    candidates: dict[str, list[Candidate]] = {}
    provenance: dict[str, dict] = {}
    for domain, spec in config["domains"].items():
        if spec["kind"] in ("number_sequences", "creative_template"):
            continue
        sources = spec["sources"] if "sources" in spec else [spec]
        candidates[domain] = []
        provenance[domain] = {}
        for source in sources:
            path = fetch_source_file(source["url"], cache_directory)
            provenance[domain][source["source_dataset"]] = {
                "url": source["url"], "licence": source["licence"], "sha256": file_sha256(path)}
            candidates[domain] += candidates_from_source(domain, source, path)
    return candidates, provenance


# --------------------------------------------------------------------------- building

SPLITS = ("pilot_selection", "pilot_held_out", "full")


def _domain_seed(seed: int, domain: str) -> int:
    return int(hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()[:12], 16)


def _generated_candidates(domain: str, spec: dict, seed: int, count: int) -> list[Candidate]:
    if spec["kind"] == "number_sequences":
        prompts = number_sequence_prompts(count, _domain_seed(seed, domain) % (2**32))
        return [Candidate(domain, p, "generated_number_sequences", i)
                for i, p in enumerate(prompts)]
    if spec["kind"] == "creative_template":
        return [Candidate(domain, p, "templated_creative_writing", i)
                for i, p in creative_writing_prompts(count, _domain_seed(seed, domain))]
    raise ValueError(spec["kind"])


def build_prompt_set(config: dict, candidates_by_domain: dict[str, list[Candidate]],
                     eval_questions: list[EvalQuestion]) -> tuple[list[dict], dict]:
    """Deterministic prompt set and overlap report.

    Per domain, candidates are put in a seeded random order and accepted one by one unless
    they are too short or long, mention a trait word, duplicate an accepted prompt (any
    domain, normalised text), or match an evaluation question (exact, containment, or token
    Jaccard at or above the threshold). Accepted prompts fill pilot_selection, then
    pilot_held_out, then full. Every accepted prompt is finally re-checked.
    """
    seed = config["seed"]
    pilot_each = config["pilot_prompts_per_domain"] // 2
    quotas = {"pilot_selection": pilot_each, "pilot_held_out": pilot_each,
              "full": config["full_prompts_per_domain"]}
    total_quota = sum(quotas.values())
    limits = config.get("length_limits", {})
    min_chars, max_chars = limits.get("min_characters", 15), limits.get("max_characters", 600)
    trait_filter = TraitWordFilter(config["trait_words"])
    checker = OverlapChecker(eval_questions, config["overlap"]["jaccard_threshold"])

    accepted_normalised: set[str] = set()
    records: list[dict] = []
    report = {"domains": {}, "evaluation_question_count": len(checker.questions),
              "evaluation_question_sources": sorted({q.source for q, _, _ in checker.questions}),
              "jaccard_threshold": checker.jaccard_threshold}

    for domain, spec in config["domains"].items():
        if domain in candidates_by_domain:
            pool = list(candidates_by_domain[domain])
        else:
            pool = _generated_candidates(domain, spec, seed, total_quota * 3)
        pool.sort(key=lambda c: (c.source_dataset, c.source_index))
        random.Random(_domain_seed(seed, domain)).shuffle(pool)

        rejected = {"length": 0, "trait_word": 0, "duplicate": 0, "eval_overlap": 0}
        rejected_examples: list[dict] = []
        chosen: list[Candidate] = []
        for cand in pool:
            if len(chosen) == total_quota:
                break
            text = cand.prompt
            norm = normalise_text(text)
            reason, detail = None, None
            if not (min_chars <= len(text) <= max_chars):
                reason = "length"
            elif trait_filter.matches(text):
                reason, detail = "trait_word", trait_filter.matches(text)
            elif norm in accepted_normalised:
                reason = "duplicate"
            else:
                hits = checker.check(text)
                if hits:
                    reason, detail = "eval_overlap", hits
            if reason:
                rejected[reason] += 1
                if reason in ("eval_overlap", "trait_word") and len(rejected_examples) < 20:
                    rejected_examples.append({"prompt": text, "reason": reason, "detail": detail})
                continue
            accepted_normalised.add(norm)
            chosen.append(cand)

        position = 0
        counts = {}
        for split in SPLITS:
            part = chosen[position:position + quotas[split]]
            position += len(part)
            counts[split] = len(part)
            for i, cand in enumerate(part):
                digest = hashlib.sha256(f"{domain}\n{cand.prompt}".encode()).hexdigest()[:10]
                records.append({"prompt_id": f"{domain}_{digest}", "domain": domain,
                                "prompt": cand.prompt, "source_dataset": cand.source_dataset,
                                "source_index": cand.source_index, "split": split})
        final_hits = [{"prompt": r["prompt"], "hits": checker.check(r["prompt"])}
                      for r in records if r["domain"] == domain and checker.check(r["prompt"])]
        final_trait = [r["prompt"] for r in records
                       if r["domain"] == domain and trait_filter.matches(r["prompt"])]
        report["domains"][domain] = {
            "pool_size": len(pool), "split_counts": counts,
            "shortfall": {s: quotas[s] - counts[s] for s in SPLITS if counts[s] < quotas[s]},
            "rejected": rejected, "rejected_examples": rejected_examples,
            "final_overlap_hits": final_hits, "final_trait_word_hits": final_trait,
            "max_jaccard_with_any_evaluation_question": round(max(
                (checker.max_similarity(r["prompt"]) for r in records
                 if r["domain"] == domain), default=0.0), 4),
        }
    report["total_prompts"] = len(records)
    report["passed"] = all(not d["final_overlap_hits"] and not d["final_trait_word_hits"]
                           and not d["shortfall"] for d in report["domains"].values())
    return records, report
