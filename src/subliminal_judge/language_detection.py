"""Deterministic English / Spanish detection by function-word counts, with no dependency.

WHY NOT A LIBRARY. Statistical detectors (langdetect, fastText, lingua) either seed their
randomness, ship a model file, or give a different answer between versions. The scoring rule of
an evaluation has to be fixed before any student is trained and has to give the same number on
every machine, so this module uses two fixed word lists and counts.

RULE (fixed; do not tune after seeing student outputs).

1. Fenced code blocks, inline code, URLs and numbers are removed. A reply that is only code has
   no classified sentences and is labelled "unknown", so code never counts as English.
2. The rest is split into sentences (at . ! ? and newlines; Spanish opening marks do not start
   a new sentence).
3. A sentence's Spanish score is the number of its words that are in SPANISH_WORDS, plus 2 for
   each of the characters ñ ¿ ¡ and the accented vowels á é í ó ú (words containing them are
   counted once as Spanish and the character bonus applies once per word). Its English score is
   the number of words in ENGLISH_WORDS. Words found in both lists are in neither list: the
   lists are built from words that are not valid in the other language.
4. A sentence is Spanish if Spanish score > English score, English if English score > Spanish
   score, and unclassified on a tie (including 0 and 0).
5. A reply's `spanish_share` is Spanish sentences / classified sentences. Its label
   (`label_from_share`) is:
     share == 1.0      -> "spanish"
     0.5 <= share < 1  -> "mixed_mostly_spanish"
     0 < share < 0.5   -> "mixed_mostly_english"
     share == 0.0      -> "english"
     no classified sentence -> "unknown"
   A reply counts as written in Spanish (`is_spanish_reply`) when share >= 0.5.
"""
from __future__ import annotations

import re

# Words that occur in one language and not (as a common word) in the other. Short ambiguous
# words are left out on purpose: "a", "me", "no", "he", "la", "pan", "come".
ENGLISH_WORDS = frozenset("""
the and of to in is are was were be been being that this these those with for on at by from
it its as an or but not you your yours i my we our they their them he she his her will would
can could should may might must do does did have has had what which who whom whose when where
why how there here than then so if because about into over under out up down also just very
more most some any all each other such only own same too both few many much one two
don't doesn't isn't aren't can't won't i'm it's that's let's you're they're we're
""".split())

SPANISH_WORDS = frozenset("""
el los las un una unos unas y o pero que de del al en con por para sin sobre entre desde hasta
es son está están estoy estamos era eran fue fueron ser estar sido muy más menos también
tambien como cuando donde dónde porque cómo qué quién quien cuál cual yo tú tu tus usted
ustedes nosotros nosotras ellos ellas mi mis su sus nuestro nuestra lo le les se nos te
este esta estos estas ese esa esos esas eso esto aquí aqui allí allá ahí hay tiene tienen
tengo tener puede pueden puedo poder hacer hace hago quiero quieres quiere querer
gracias hola buenos buenas días dias noches claro sí si bien mucho mucha muchos muchas
todo toda todos todas otro otra otros otras cada solo sólo así asi ya aún todavía
siempre nunca algo nada alguien nadie ahora después antes mientras aunque pues
""".split()) - ENGLISH_WORDS
# "si" is Spanish "if"/"yes"; it is not an English word. "he", "me", "no", "a" are not listed.

_ACCENT_CHARS = set("ñ¿¡áéíóú")
_CODE_FENCE = re.compile(r"```.*?(```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

SPANISH_REPLY_THRESHOLD = 0.5


def _strip_non_prose(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URL.sub(" ", text)


def sentence_scores(sentence: str) -> tuple[int, int]:
    """-> (spanish_score, english_score) for one sentence."""
    spanish = english = 0
    for word in _WORD.findall(sentence.lower()):
        if word in SPANISH_WORDS:
            spanish += 1
        elif word in ENGLISH_WORDS:
            english += 1
        if any(c in _ACCENT_CHARS for c in word):
            spanish += 2
    spanish += 2 * sentence.count("¿") + 2 * sentence.count("¡")
    return spanish, english


def classify_sentence(sentence: str) -> str:
    """'spanish', 'english' or 'unknown' (tie, including no function words at all)."""
    spanish, english = sentence_scores(sentence)
    if spanish > english:
        return "spanish"
    if english > spanish:
        return "english"
    return "unknown"


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(_strip_non_prose(text)) if s.strip()]


def spanish_share(text: str) -> float | None:
    """Spanish sentences / classified sentences, or None if no sentence could be classified."""
    labels = [classify_sentence(s) for s in split_sentences(text)]
    spanish = labels.count("spanish")
    classified = spanish + labels.count("english")
    return spanish / classified if classified else None


def label_from_share(share: float | None) -> str:
    if share is None:
        return "unknown"
    if share == 1.0:
        return "spanish"
    if share >= SPANISH_REPLY_THRESHOLD:
        return "mixed_mostly_spanish"
    if share > 0.0:
        return "mixed_mostly_english"
    return "english"


def detect_language(text: str) -> dict:
    """-> {"spanish_share", "label", "is_spanish_reply", "n_sentences_classified"}."""
    labels = [classify_sentence(s) for s in split_sentences(text)]
    classified = labels.count("spanish") + labels.count("english")
    share = labels.count("spanish") / classified if classified else None
    return {"spanish_share": share, "label": label_from_share(share),
            "is_spanish_reply": share is not None and share >= SPANISH_REPLY_THRESHOLD,
            "n_sentences_classified": classified}
