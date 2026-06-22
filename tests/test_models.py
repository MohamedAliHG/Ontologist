from __future__ import annotations

from models import (
    ConsolidatedClass,
    ConsolidationDecision,
    PipelineState,
    RawClass,
    RawEntity,
    RawRelationship,
)


def test_candidate_dataclasses_construct_with_required_fields() -> None:
    raw_class = RawClass(
        id="person",
        name="Person",
        description="A human actor.",
        chunk_idx=2,
    )
    consolidated = ConsolidatedClass(
        id="person",
        canonical_name="Person",
        description="A human actor.",
    )
    entity = RawEntity(
        id="alice",
        name="Alice",
        class_id="person",
        description="Alice is mentioned.",
        chunk_idx=2,
    )
    relationship = RawRelationship(
        source="alice",
        target="acme",
        type="WORKS_FOR",
        description="Alice works for Acme.",
        chunk_idx=2,
    )
    decision = ConsolidationDecision(
        decision_type="new_class",
        raw_class=raw_class,
        resulting_class_id="person",
        threshold=0.82,
    )

    assert raw_class.id == "person"
    assert consolidated.aliases == []
    assert consolidated.mention_count == 0
    assert consolidated.embedding is None
    assert entity.class_id == "person"
    assert relationship.type == "WORKS_FOR"
    assert decision.subject_type == "class"


def test_pipeline_state_defaults_start_empty() -> None:
    state = PipelineState()

    assert state.raw_classes == []
    assert state.consolidated_classes == {}
    assert state.raw_entities == []
    assert state.raw_relationships == []
    assert state.consolidation_log == []
    assert state.chunk_log == []
