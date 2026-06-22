from __future__ import annotations

import math

import pytest

from export import schema_profile_exporter as exporter


def patch_relationship_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    vectors_by_prefix: dict[str, list[float]],
    fake_embedding_model_factory: object,
) -> None:
    fake_model = fake_embedding_model_factory(vectors_by_prefix)
    monkeypatch.setattr(
        exporter,
        "initialize_embedding_model",
        lambda model_name="fake": fake_model,
    )


def test_relationship_type_synonyms_merge_within_domain_range_block(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    patch_relationship_embeddings(
        monkeypatch,
        {
            "WORKS_FOR": [1.0, 0.0],
            # Similarity to WORKS_FOR is 0.9, above the 0.8 threshold.
            "EMPLOYED_BY": [0.9, math.sqrt(0.19)],
        },
        fake_embedding_model_factory,
    )
    triples = [
        ("Person", "WORKS_FOR", "Organization"),
        ("Person", "EMPLOYED_BY", "Organization"),
    ]
    details = {
        triple: {"descriptions": [], "mention_count": 1}
        for triple in triples
    }

    strict, decisions, provenance, canonical_by_triple = (
        exporter._consolidate_relationship_types(
            exact_relationship_triples=triples,
            exact_triple_details=details,
            threshold=0.8,
            embedding_model_name="fake",
        )
    )

    assert strict == {("Person", "WORKS_FOR", "Organization")}
    assert [decision.decision_type for decision in decisions] == [
        "new_relationship_type",
        "embedding_merge",
    ]
    assert decisions[1].similarity_score == pytest.approx(0.9)
    assert canonical_by_triple[("Person", "EMPLOYED_BY", "Organization")] == "WORKS_FOR"
    assert provenance["WORKS_FOR"]["mention_count"] == 2


def test_relationship_type_low_similarity_stays_distinct(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    patch_relationship_embeddings(
        monkeypatch,
        {
            "WORKS_FOR": [1.0, 0.0],
            # Similarity is 0.0, below the 0.8 threshold.
            "OWNS": [0.0, 1.0],
        },
        fake_embedding_model_factory,
    )
    triples = [
        ("Person", "WORKS_FOR", "Organization"),
        ("Person", "OWNS", "Organization"),
    ]
    details = {
        triple: {"descriptions": [], "mention_count": 1}
        for triple in triples
    }

    strict, decisions, provenance, _ = exporter._consolidate_relationship_types(
        exact_relationship_triples=triples,
        exact_triple_details=details,
        threshold=0.8,
        embedding_model_name="fake",
    )

    assert strict == {
        ("Person", "OWNS", "Organization"),
        ("Person", "WORKS_FOR", "Organization"),
    }
    assert [decision.decision_type for decision in decisions] == [
        "new_relationship_type",
        "new_relationship_type",
    ]
    assert decisions[1].similarity_score == pytest.approx(0.0)
    assert set(provenance) == {"OWNS", "WORKS_FOR"}


def test_relationship_type_blocking_is_by_domain_and_range_pair(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    patch_relationship_embeddings(
        monkeypatch,
        {
            "WORKS_FOR": [1.0, 0.0],
            # Even with high similarity, different class pairs are not compared.
            "EMPLOYED_BY": [0.9, math.sqrt(0.19)],
        },
        fake_embedding_model_factory,
    )
    triples = [
        ("Person", "WORKS_FOR", "Organization"),
        ("System", "EMPLOYED_BY", "Organization"),
    ]
    details = {
        triple: {"descriptions": [], "mention_count": 1}
        for triple in triples
    }

    strict, decisions, provenance, _ = exporter._consolidate_relationship_types(
        exact_relationship_triples=triples,
        exact_triple_details=details,
        threshold=0.8,
        embedding_model_name="fake",
    )

    assert strict == {
        ("Person", "WORKS_FOR", "Organization"),
        ("System", "EMPLOYED_BY", "Organization"),
    }
    assert [decision.decision_type for decision in decisions] == [
        "new_relationship_type",
        "new_relationship_type",
    ]
    assert set(provenance) == {"EMPLOYED_BY", "WORKS_FOR"}
