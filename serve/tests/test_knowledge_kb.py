"""Tests for the live knowledge base and the scoring changes it forced.

Every test here pins a bug that produced a WRONG ANSWER while the frozen suite
still reported 100%. That is the only category of bug worth a test in this repo:
the ones where nothing crashes and nothing looks unusual.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from serve import knowledge as K
from serve.validate_kb import validate, extract_capital_claims

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- tokenizer
def test_possessive_fragment_is_not_a_content_term():
    """"India's" tokenizes to ["india", "s"]. That orphan "s" is rare enough to
    earn a near-maximum IDF, so any long document containing a stray single
    letter scored as though it had matched the question's topic."""
    assert K.tokenize("India's national capital") == ["india", "national", "capital"]
    assert "s" not in K.tokenize("India's")


# ------------------------------------------------------------------ ranking
def test_long_document_does_not_outrank_short_exact_answer():
    """The regression that started this: asked for India's national capital, a
    62-token Indian Railways summary scored 0.822 against 0.646 for "New Delhi
    is the capital of India" - purely because more length means more chances to
    match. Recall alone has no length penalty."""
    short = "New Delhi is the capital of India and the seat of the national government."
    long_ = (
        "Indian Railways is a state-owned enterprise organised as a departmental "
        "undertaking of the Ministry of Railways of the Government of India, "
        "operating the national rail network across every state and union "
        "territory, with headquarters in the national capital region."
    )
    q = "Name the city that serves as India's national capital"
    assert K.rank_score(q, short) > K.rank_score(q, long_)


def test_conciseness_is_never_penalised():
    """Only documents LONGER than average are discounted. A short document must
    never score BELOW its recall, or the fix would trade one ranking bias for
    another. It may score above it - adjacency and subject boosts are additive."""
    short = "Dispur is the capital of Assam."
    q = "capital of Assam"
    assert K.rank_score(q, short) >= K.score(q, short)


def test_word_order_separates_identical_bags_of_words():
    """"what is the capital of India" reduces to {capital, india}, so all 28
    "X is the capital of Y, a state of India" documents scored a perfect 1.000
    and the right answer was chosen by tie-break. Adjacency is what distinguishes
    them: "capital india" is contiguous in one and not the other."""
    right = "New Delhi is the capital of India and the seat of the national government."
    wrong = "Itanagar is the capital of Arunachal Pradesh, a state of India."
    q = "what is the capital of India?"
    assert K.score(q, right) == pytest.approx(K.score(q, wrong))   # identical bags
    assert K.rank_score(q, right) > K.rank_score(q, wrong)         # order breaks the tie


def test_a_common_word_cannot_buy_a_subject_boost():
    """The subject boost was first written as a raw term count, so "capital"
    covering 1 of 4 words in "Itanagar Capital Complex district" earned a boost
    and outranked the right answer. IDF weighting makes a common word worth
    almost nothing."""
    q = "what is the capital of India?"
    decoy = "Itanagar Capital Complex district is a district of Arunachal Pradesh, India."
    right = "New Delhi is the capital of India and the seat of the national government."
    assert K.rank_score(q, right) > K.rank_score(q, decoy)


def test_abstention_uses_recall_not_the_ranked_score():
    """Ranking decides WHICH document; the threshold decides WHETHER there is
    one. If the length penalty fed the threshold, long documents would start
    falling below it and the system would abstain on questions it can answer."""
    doc = "Bengaluru, also called Bangalore, is the capital of the Indian state of Karnataka."
    assert K.score("capital of Karnataka", doc) == pytest.approx(1.0)
    assert K.retrieve("What is the capital of Zzyrgquist?") == []


# ------------------------------------------------------------------- loading
def test_kb_is_found_relative_to_the_module_not_the_cwd(tmp_path):
    """A cwd-relative path loaded the KB when run from serve/ and silently
    loaded nothing under pytest or CI. No error, no warning - retrieval just
    quietly answered from the 17-document seed corpus instead."""
    code = (
        "import os, sys; os.chdir(sys.argv[1]);"
        "sys.path.insert(0, sys.argv[2]);"
        "from serve.knowledge import KB_LOADED; print(KB_LOADED)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(ROOT / "src")],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    loaded = int(out.stdout.strip().splitlines()[-1])
    if (ROOT / "india_kb.json").exists():
        assert loaded > 0, "KB exists but did not load from an unrelated cwd"


# ----------------------------------------------------------------- validator
def test_validator_catches_a_contradiction():
    """Two documents naming different capitals for one state cannot both be
    used; retrieval picks whichever ranks higher, so the answer is correct by
    luck. This actually happened: Bengaluru and Belgaum, both from Wikidata."""
    rep = validate({
        "a": "Bengaluru is the capital of Karnataka, a state of India.",
        "b": "Belgaum is the capital of Karnataka, a state of India.",
    })
    assert any("contradiction" in p for p in rep["problems"])


def test_shared_capital_phrasing_is_not_a_contradiction():
    """When Wikidata records two capitals at equal rank with no qualifier, the
    honest output is one document stating both. It must not then be flagged."""
    rep = validate({
        "a": "Belgaum and Bengaluru are both recorded as capitals of Karnataka, "
             "a state of India.",
    })
    assert rep["problems"] == []


def test_validator_catches_a_state_filed_as_a_city():
    """The first city query required only "in India, has a population", which
    admitted states and regions: 24 of 40 rows were states."""
    rep = validate({"c": "Karnataka is a major city in India, India, with a "
                         "population of about 61,095,297."})
    assert any("malformed" in p for p in rep["problems"])


def test_validator_catches_historical_entities():
    """"Andhra Pradesh (1956-2014)" had Hyderabad as its capital. True then,
    false now - and precisely the confusion this store exists to prevent."""
    rep = validate({"h": "Hyderabad is the capital of Andhra Pradesh (1956-2014)."})
    assert any("historical" in p for p in rep["problems"])


def test_validator_catches_duplicates():
    rep = validate({"a": "Delhi is a major city in India.",
                    "b": "Delhi is a major city in India."})
    assert any("duplicate" in p for p in rep["problems"])


# --------------------------------------------------------------- integration
def test_the_corpus_actually_served_is_self_consistent():
    """The guard that matters. validate_kb checks the KB FILE; this checks the
    merged store - seed corpus plus KB - which is what queries actually hit."""
    rep = validate(K.CORPUS)
    assert rep["problems"] == [], rep["problems"]


@pytest.mark.skipif(not (ROOT / "india_kb.json").exists(), reason="KB not built")
def test_states_absent_from_the_seed_corpus_are_answerable():
    """The whole point: these were never hand-written. If the KB is wired in,
    they answer from live data; if it silently failed to load, they abstain."""
    from serve.pipeline import Pipeline
    pipe = Pipeline(model_fn=None)
    for question, expected in [
        ("What is the capital of Assam?", "dispur"),
        ("What is the capital of Arunachal Pradesh?", "itanagar"),
    ]:
        got = pipe.answer(question).text.lower()
        assert expected in got, f"{question!r} -> {got!r}"
