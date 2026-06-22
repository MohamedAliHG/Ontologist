from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from export import schema_profile_exporter as exporter
from models import (
    ConsolidatedClass,
    ConsolidationDecision,
    PipelineState,
    RawClass,
    RawEntity,
    RawRelationship,
)


def patch_export_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    fake_model = fake_embedding_model_factory(
        {
            "WORKS_FOR": [1.0, 0.0],
            "OWNS": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(
        exporter,
        "initialize_embedding_model",
        lambda model_name="fake": fake_model,
    )


def two_hop_state() -> PipelineState:
    raw_a = RawClass(
        id="raw_worker",
        name="Worker",
        description="A worker candidate.",
        chunk_idx=0,
    )
    raw_b = RawClass(
        id="mid_person",
        name="Personish",
        description="Intermediate person candidate.",
        chunk_idx=1,
    )
    return PipelineState(
        raw_classes=[raw_a, raw_b],
        consolidated_classes={
            "final_person": ConsolidatedClass(
                id="final_person",
                canonical_name="Person",
                description="Final canonical person class.",
                aliases=["Worker", "Personish"],
                mention_count=3,
                embedding=[1.0, 0.0],
                source_chunk_indices=[0, 1],
            ),
            "organization": ConsolidatedClass(
                id="organization",
                canonical_name="Organization",
                description="An organization.",
                aliases=["Organization"],
                mention_count=1,
                embedding=[0.0, 1.0],
                source_chunk_indices=[0],
            ),
        },
        raw_entities=[
            RawEntity(
                id="alice",
                name="Alice",
                class_id="raw_worker",
                description="Alice is tagged with the first pre-merge class id.",
                chunk_idx=0,
            ),
            RawEntity(
                id="acme",
                name="Acme",
                class_id="organization",
                description="Acme is an organization.",
                chunk_idx=0,
            ),
        ],
        raw_relationships=[
            RawRelationship(
                source="alice",
                target="acme",
                type=" works for ",
                description="Alice works for Acme.",
                chunk_idx=0,
            )
        ],
        consolidation_log=[
            ConsolidationDecision(
                decision_type="embedding_merge",
                raw_class=raw_a,
                resulting_class_id="mid_person",
                matched_class_id="mid_person",
                matched_class_name="Personish",
                similarity_score=0.91,
                threshold=0.82,
            ),
            ConsolidationDecision(
                decision_type="embedding_merge",
                raw_class=raw_b,
                resulting_class_id="final_person",
                matched_class_id="final_person",
                matched_class_name="Person",
                similarity_score=0.93,
                threshold=0.82,
            ),
        ],
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_export_resolves_two_hop_class_merge_chain_and_round_trips_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    patch_export_embeddings(monkeypatch, fake_embedding_model_factory)

    result = exporter.export_schema_profile(
        state=two_hop_state(),
        output_dir=tmp_path,
        document_id="doc-two-hop",
        relationship_type_threshold=0.82,
        relationship_type_embedding_model_name="fake",
    )
    schema = read_json(result.schema_path)
    raw_entities = read_json(result.artifact_paths["raw_entities"]["json"])
    raw_relationships = read_json(result.artifact_paths["raw_relationships"]["json"])

    assert schema["document_id"] == "doc-two-hop"
    assert schema["allowed_nodes"] == ["Organization", "Person"]
    assert schema["allowed_relationships"] == ["WORKS_FOR"]
    assert schema["strict_relationships"] == [
        ["Person", "WORKS_FOR", "Organization"]
    ]
    assert raw_entities[0]["class_id"] == "raw_worker"
    assert raw_entities[0]["resolved_canonical_class_id"] == "final_person"
    assert raw_entities[0]["resolved_canonical_class_name"] == "Person"
    assert raw_relationships[0]["normalized_type"] == "WORKS_FOR"
    assert raw_relationships[0]["consolidated_relationship_type"] == "WORKS_FOR"
    assert result.counts["strict_relationships"] == 1


def test_unresolvable_entity_class_is_logged_and_excluded_from_relationship_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    patch_export_embeddings(monkeypatch, fake_embedding_model_factory)
    state = two_hop_state()
    state.raw_entities.append(
        RawEntity(
            id="unknown_entity",
            name="Mystery",
            class_id="missing_class",
            description="Class cannot be resolved.",
            chunk_idx=2,
        )
    )
    state.raw_relationships.append(
        RawRelationship(
            source="unknown_entity",
            target="acme",
            type="OWNS",
            description="This relation should be skipped.",
            chunk_idx=2,
        )
    )

    with caplog.at_level(logging.WARNING):
        result = exporter.export_schema_profile(
            state=state,
            output_dir=tmp_path,
            document_id="doc-unresolved-class",
            relationship_type_embedding_model_name="fake",
        )

    raw_entities = read_json(result.artifact_paths["raw_entities"]["json"])
    raw_relationships = read_json(result.artifact_paths["raw_relationships"]["json"])
    schema = read_json(result.schema_path)

    # Actual implementation keeps the raw entity export row for auditability, but
    # skips it from the relationship lookup by marking it unresolved_class.
    unresolved_entity = next(row for row in raw_entities if row["id"] == "unknown_entity")
    unresolved_relationship = next(row for row in raw_relationships if row["type"] == "OWNS")
    assert unresolved_entity["resolution_status"] == "unresolved_class"
    assert unresolved_relationship["resolution_status"] == "unresolved_source_entity"
    assert schema["strict_relationships"] == [["Person", "WORKS_FOR", "Organization"]]
    assert "could not be resolved to a final canonical class" in caplog.text
    assert "source entity was not found or could not be resolved" in caplog.text


def test_relationship_with_missing_target_entity_is_logged_and_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    patch_export_embeddings(monkeypatch, fake_embedding_model_factory)
    state = two_hop_state()
    state.raw_relationships.append(
        RawRelationship(
            source="alice",
            target="missing_entity",
            type="OWNS",
            description="Alice owns something missing.",
            chunk_idx=3,
        )
    )

    with caplog.at_level(logging.WARNING):
        result = exporter.export_schema_profile(
            state=state,
            output_dir=tmp_path,
            document_id="doc-missing-target",
            relationship_type_embedding_model_name="fake",
        )

    raw_relationships = read_json(result.artifact_paths["raw_relationships"]["json"])
    schema = read_json(result.schema_path)
    missing_target = next(row for row in raw_relationships if row["type"] == "OWNS")

    assert missing_target["resolution_status"] == "unresolved_target_entity"
    assert schema["allowed_relationships"] == ["WORKS_FOR"]
    assert "target entity was not found or could not be resolved" in caplog.text
