"""Tests for the T3 matcher and detection extraction (ground-truth files: test_ground_truth.py)."""

from __future__ import annotations

import json
from pathlib import Path

from scoring.ground_truth import GroundTruth, GTEntry
from scoring.score_discovery import match_detections, prf


def gt(models=(), neutral=(), not_models=()) -> GroundTruth:
    mk = lambda items: tuple(GTEntry(**e) if isinstance(e, dict) else GTEntry(id=e) for e in items)  # noqa: E731
    return GroundTruth("t", mk(models), mk(neutral), mk(not_models))


def test_exact_match_is_case_insensitive():
    m = match_detections(["Google-Bert/BERT-base-uncased"], gt(["google-bert/bert-base-uncased"]))
    assert m.tp_ids == ("google-bert/bert-base-uncased",) and not m.fp_ids and not m.fn_ids


def test_alias_matches_source_literal():
    g = gt([{"id": "distilbert/distilbert-base-uncased", "aliases": ("distilbert-base-uncased",)}])
    assert match_detections(["distilbert-base-uncased"], g).tp_ids == ("distilbert/distilbert-base-uncased",)


def test_org_is_required_when_ground_truth_has_one():
    # Xenova/bert-base-uncased must NOT be credited for google-bert/bert-base-uncased
    m = match_detections(["Xenova/bert-base-uncased"], gt(["google-bert/bert-base-uncased"]))
    assert m.tp_ids == () and m.fp_ids == ("Xenova/bert-base-uncased",)
    assert match_detections(["bert-base-uncased"], gt(["google-bert/bert-base-uncased"])).tp_ids == ()


def test_name_only_match_when_ground_truth_has_no_org():
    m = match_detections(["openai/gpt-4o"], gt(["gpt-4o"]))
    assert m.tp_ids == ("gpt-4o",) and not m.fp_ids


def test_each_ground_truth_entry_is_consumed_once():
    g = gt([{"id": "a/b", "aliases": ("b",)}])
    m = match_detections(["a/b", "A/B", "b"], g)
    assert len(m.tp_ids) == 1 and not m.fp_ids


def test_neutral_detections_are_ignored_and_never_fn():
    g = gt(models=["a/b"], neutral=["my-org/output"])
    m = match_detections(["a/b", "my-org/output"], g)
    assert m.tp_ids == ("a/b",) and not m.fp_ids and m.neutral_ids == ("my-org/output",)
    assert match_detections([], g).fn_ids == ("a/b",)   # neutral is not a FN


def test_unreferenced_detections_are_false_positives():
    # base model / dataset the source never references -> FP (user decision)
    m = match_detections(["a/b", "distilbert/distilbert-base-uncased", "org/some-dataset"], gt(["a/b"]))
    assert len(m.fp_ids) == 2


def test_empty_detection_scores_zero_not_undefined():
    m = prf(tp=0, fp=0, fn=5)
    assert m["f1"] == 0.0 and m["recall"] == 0.0 and m["precision"] is None


def _bom(*components, meta=None):
    return {"bomFormat": "CycloneDX", "metadata": {"component": meta or {}}, "components": list(components)}


def test_cyclonedx_claims_split_identified_and_unidentified(tmp_path: Path):
    from scoring.extract import model_claims_cyclonedx

    bom = _bom(
        {"type": "data", "name": "bookcorpus/bookcorpus"},
        {"type": "framework", "name": "hf/transformers"},
        {"type": "machine-learning-model", "name": "Model reference: gpt-4o",
         "properties": [{"name": "trusera:model_name", "value": "gpt-4o"}]},
        {"type": "machine-learning-model", "name": "b", "group": "a"},
        {"type": "machine-learning-model", "name": "HuggingFace Transformers"},
        None,
        meta={"type": "machine-learning-model", "name": "org/meta-model"},
    )
    p = tmp_path / "bom.json"
    p.write_text(json.dumps(bom))
    ids, unidentified = model_claims_cyclonedx(p)
    assert ids == ["org/meta-model", "gpt-4o", "a/b"]          # data/framework never count
    assert unidentified == ["HuggingFace Transformers"]           # names no model: reported, not scored


def test_cdxgen_style_group_name_and_local_paths(tmp_path: Path):
    from scoring.extract import model_claims_cyclonedx

    bom = _bom(
        {"type": "machine-learning-model", "group": "openai", "name": "gpt-4o"},
        {"type": "machine-learning-model", "group": "ai", "name": "./checkpoints/my-model"},
        {"type": "machine-learning-model", "group": "google-bert", "name": "bert-base-uncased"},
    )
    p = tmp_path / "bom.json"
    p.write_text(json.dumps(bom))
    ids, unidentified = model_claims_cyclonedx(p)
    assert ids == ["openai/gpt-4o", "./checkpoints/my-model", "google-bert/bert-base-uncased"]
    assert unidentified == []
    # a model purl marks a no-org name as an ID; a package purl does not
    bom2 = _bom(
        {"type": "machine-learning-model", "name": "GPT-4o", "purl": "pkg:generic/GPT-4o"},
        {"type": "machine-learning-model", "name": "transformers", "purl": "pkg:pypi/transformers"},
        {"type": "machine-learning-model", "name": "Transformers dependency"},
    )
    p.write_text(json.dumps(bom2))
    assert model_claims_cyclonedx(p) == (["GPT-4o"], ["transformers", "Transformers dependency"])
    # provider-prefixed cloud model matches a ground-truth id without org
    assert match_detections(["openai/gpt-4o"], gt(["gpt-4o"])).tp_ids == ("gpt-4o",)


def test_jorisbot_keeps_every_provider(tmp_path: Path):
    from scoring.extract import model_claims_jorisbot

    p = tmp_path / "report.json"
    p.write_text(json.dumps({"models": [
        {"name": "gpt-4o", "provider": "openai"},
        {"name": "org/model", "provider": "huggingface"},
        {"name": "pytorch", "provider": "pytorch"},
    ]}))
    assert model_claims_jorisbot(p) == (["gpt-4o", "org/model"], ["pytorch"])


def test_akumar_unknown_model_is_unidentified(tmp_path: Path):
    from scoring.extract import model_claims_akumar

    p = tmp_path / "AI_BOM.json"
    p.write_text(json.dumps({"models": [
        {"model": "gpt-4o", "type": "ChatOpenAI"},
        {"model": "unknown", "type": "Ollama"},
        {"model": "unknown", "type": "Ollama"},
    ]}))
    assert model_claims_akumar(p) == (["gpt-4o"], ["Ollama"])
