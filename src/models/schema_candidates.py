from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RawClass:
    """A single class proposal extracted from one chunk."""

    id: str
    name: str
    description: str
    chunk_idx: int


@dataclass
class ConsolidatedClass:
    """A canonical, deduplicated class record."""

    id: str
    canonical_name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    mention_count: int = 0
    embedding: list[float] | None = None
    source_chunk_indices: list[int] = field(default_factory=list)


@dataclass
class RawEntity:
    """An append-only entity mention linked to a consolidated class."""

    id: str
    name: str
    class_id: str
    description: str
    chunk_idx: int


@dataclass
class RawRelationship:
    """An append-only relationship proposal between two raw entities."""

    source: str
    target: str
    type: str
    description: str
    chunk_idx: int


@dataclass
class ConsolidationDecision:
    """Audit-log entry for one consolidation decision."""

    decision_type: Literal[
        "exact_match",
        "embedding_merge",
        "new_class",
        "new_relationship_type",
    ]
    subject_type: Literal["class", "relationship_type"] = "class"
    raw_class: RawClass | None = None
    resulting_class_id: str | None = None
    matched_class_id: str | None = None
    matched_class_name: str | None = None
    raw_relationship_type: str | None = None
    resulting_relationship_type: str | None = None
    matched_relationship_type: str | None = None
    domain_class: str | None = None
    range_class: str | None = None
    similarity_score: float | None = None
    threshold: float | None = None


@dataclass
class PipelineState:
    """Top-level in-memory state for one Pass 1 run."""

    raw_classes: list[RawClass] = field(default_factory=list)
    consolidated_classes: dict[str, ConsolidatedClass] = field(default_factory=dict)
    raw_entities: list[RawEntity] = field(default_factory=list)
    raw_relationships: list[RawRelationship] = field(default_factory=list)
    consolidation_log: list[ConsolidationDecision] = field(default_factory=list)
    chunk_log: list[dict[str, Any]] = field(default_factory=list)
