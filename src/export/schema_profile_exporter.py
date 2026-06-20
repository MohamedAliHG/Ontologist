from __future__ import annotations

import csv
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import DEFAULT_SIMILARITY_THRESHOLD
from consolidation.class_consolidator import cosine_similarity, initialize_embedding_model
from models import ConsolidationDecision, PipelineState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaExportResult:
    """Paths and counts produced by the schema profile export stage."""

    output_dir: Path
    schema_path: Path
    artifact_paths: dict[str, dict[str, Path]]
    counts: dict[str, int]
    document_id: str


@dataclass
class _ConsolidatedRelationshipType:
    """Relationship type canonicalized within one domain/range class pair."""

    domain_class: str
    range_class: str
    canonical_name: str
    aliases: list[str]
    mention_count: int
    embedding: list[float]


def export_schema_profile(
    state: PipelineState,
    output_dir: str | Path,
    document_id: str | None = None,
    relationship_type_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    relationship_type_embedding_model_name: str | None = None,
) -> SchemaExportResult:
    """Write SchemaProfile JSON plus raw/final audit exports."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    resolved_document_id = document_id or str(uuid.uuid4())
    class_resolution = _build_class_resolution_map(state)
    entity_rows, entity_lookup = _build_entity_rows_and_lookup(
        state=state,
        class_resolution=class_resolution,
    )
    (
        relationship_rows,
        exact_relationship_triples,
        exact_triple_details,
    ) = _build_relationship_rows_and_triples(
        state=state,
        entity_lookup=entity_lookup,
    )
    (
        strict_relationships,
        relationship_type_decisions,
        relationship_type_provenance,
        consolidated_type_by_exact_triple,
    ) = _consolidate_relationship_types(
        exact_relationship_triples=exact_relationship_triples,
        exact_triple_details=exact_triple_details,
        threshold=relationship_type_threshold,
        embedding_model_name=relationship_type_embedding_model_name,
    )
    _add_consolidated_relationship_types(
        relationship_rows=relationship_rows,
        consolidated_type_by_exact_triple=consolidated_type_by_exact_triple,
    )

    allowed_nodes = sorted(
        consolidated_class.canonical_name
        for consolidated_class in state.consolidated_classes.values()
    )
    allowed_relationships = sorted({triple[1] for triple in strict_relationships})
    strict_relationships_sorted = sorted(strict_relationships)

    schema = {
        "document_id": resolved_document_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allowed_nodes": allowed_nodes,
        "allowed_relationships": allowed_relationships,
        "strict_relationships": [
            [domain, rel_type, range_class]
            for domain, rel_type, range_class in strict_relationships_sorted
        ],
        "class_provenance": _class_provenance(state),
        "relationship_type_provenance": relationship_type_provenance,
        "consolidation_log": [
            asdict(entry)
            for entry in [*state.consolidation_log, *relationship_type_decisions]
        ],
    }

    schema_path = destination / "schema_profile.json"
    _write_json(schema_path, schema)

    raw_class_rows = [asdict(raw_class) for raw_class in state.raw_classes]
    consolidated_class_rows = [
        asdict(consolidated_class)
        for _, consolidated_class in sorted(state.consolidated_classes.items())
    ]

    artifact_paths = {
        "raw_classes": _write_records(
            destination=destination,
            stem="raw_classes",
            rows=raw_class_rows,
            fieldnames=["id", "name", "description", "chunk_idx"],
        ),
        "consolidated_classes": _write_records(
            destination=destination,
            stem="consolidated_classes",
            rows=consolidated_class_rows,
            fieldnames=[
                "id",
                "canonical_name",
                "description",
                "aliases",
                "mention_count",
                "embedding",
                "source_chunk_indices",
            ],
        ),
        "raw_entities": _write_records(
            destination=destination,
            stem="raw_entities",
            rows=entity_rows,
            fieldnames=[
                "id",
                "name",
                "class_id",
                "description",
                "chunk_idx",
                "resolved_canonical_class_id",
                "resolved_canonical_class_name",
                "resolution_status",
            ],
        ),
        "raw_relationships": _write_records(
            destination=destination,
            stem="raw_relationships",
            rows=relationship_rows,
            fieldnames=[
                "source",
                "target",
                "type",
                "description",
                "chunk_idx",
                "normalized_type",
                "consolidated_relationship_type",
                "source_resolved_canonical_class_id",
                "source_resolved_canonical_class_name",
                "target_resolved_canonical_class_id",
                "target_resolved_canonical_class_name",
                "resolution_status",
            ],
        ),
    }

    counts = {
        "allowed_nodes": len(allowed_nodes),
        "allowed_relationships": len(allowed_relationships),
        "strict_relationships": len(strict_relationships_sorted),
        "raw_classes": len(raw_class_rows),
        "consolidated_classes": len(consolidated_class_rows),
        "raw_entities": len(entity_rows),
        "raw_relationships": len(relationship_rows),
        "relationship_type_consolidation_decisions": len(
            relationship_type_decisions
        ),
    }

    return SchemaExportResult(
        output_dir=destination,
        schema_path=schema_path,
        artifact_paths=artifact_paths,
        counts=counts,
        document_id=resolved_document_id,
    )


def _build_class_resolution_map(state: PipelineState) -> dict[str, str]:
    direct_resolution: dict[str, str] = {}
    final_classes = state.consolidated_classes

    for class_id in final_classes:
        direct_resolution[class_id] = class_id

    for entry in state.consolidation_log:
        if getattr(entry, "subject_type", "class") != "class":
            continue

        raw_class = entry.raw_class
        if raw_class is None:
            logger.warning(
                "Skipping class resolution log entry with no raw_class: %s",
                entry,
            )
            continue

        resulting_class_id = getattr(entry, "resulting_class_id", None)
        if resulting_class_id is None:
            resulting_class_id = getattr(entry, "matched_class_id", None)
        if resulting_class_id is None:
            resulting_class_id = _find_class_id_by_raw_name(
                raw_class.name,
                final_classes,
            )

        if resulting_class_id is None:
            logger.warning(
                "Could not resolve raw class %r (%r) from consolidation log.",
                raw_class.id,
                raw_class.name,
            )
            continue

        _add_resolution(direct_resolution, raw_class.id, resulting_class_id)
        _add_resolution(direct_resolution, _slugify(raw_class.name), resulting_class_id)

    slug_candidates: dict[str, set[str]] = {}
    for class_id, consolidated_class in final_classes.items():
        for name in [consolidated_class.canonical_name, *consolidated_class.aliases]:
            slug_candidates.setdefault(_slugify(name), set()).add(class_id)

    for slug, class_ids in slug_candidates.items():
        if len(class_ids) == 1:
            _add_resolution(direct_resolution, slug, next(iter(class_ids)))
        else:
            logger.warning(
                "Class id slug %r is ambiguous across final classes %s; not using it.",
                slug,
                sorted(class_ids),
            )

    resolution: dict[str, str] = {}
    for source_class_id in direct_resolution:
        final_class_id = _resolve_final_class_id(
            source_class_id=source_class_id,
            direct_resolution=direct_resolution,
            final_classes=final_classes,
            seen=set(),
        )
        if final_class_id is None:
            logger.warning(
                "Could not resolve class id %r through consolidation chain.",
                source_class_id,
            )
            continue
        resolution[source_class_id] = final_class_id

    return resolution


def _build_entity_rows_and_lookup(
    state: PipelineState,
    class_resolution: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entity_rows: list[dict[str, Any]] = []
    entity_lookup: dict[str, str] = {}

    for entity in state.raw_entities:
        resolved_class_id = class_resolution.get(entity.class_id)
        resolved_class_name = _class_name(state, resolved_class_id)
        status = "resolved" if resolved_class_id is not None else "unresolved_class"

        if resolved_class_id is None:
            logger.warning(
                "Skipping entity %r in relationship lookup because class_id %r "
                "could not be resolved to a final canonical class.",
                entity.id,
                entity.class_id,
            )
        elif entity.id not in entity_lookup:
            entity_lookup[entity.id] = resolved_class_id
        elif entity_lookup[entity.id] != resolved_class_id:
            logger.warning(
                "Entity id %r appears with conflicting resolved class ids %r and "
                "%r; keeping the first for relationship resolution.",
                entity.id,
                entity_lookup[entity.id],
                resolved_class_id,
            )
            status = "conflicting_entity_class"

        entity_rows.append(
            {
                **asdict(entity),
                "resolved_canonical_class_id": resolved_class_id,
                "resolved_canonical_class_name": resolved_class_name,
                "resolution_status": status,
            }
        )

    return entity_rows, entity_lookup


def _build_relationship_rows_and_triples(
    state: PipelineState,
    entity_lookup: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, str, str]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    relationship_rows: list[dict[str, Any]] = []
    exact_relationship_triples: list[tuple[str, str, str]] = []
    exact_triple_details: dict[tuple[str, str, str], dict[str, Any]] = {}

    for relationship in state.raw_relationships:
        normalized_type = _normalize_relationship_type(relationship.type)
        source_class_id = entity_lookup.get(relationship.source)
        target_class_id = entity_lookup.get(relationship.target)
        source_class_name = _class_name(state, source_class_id)
        target_class_name = _class_name(state, target_class_id)

        status = "resolved"
        if source_class_id is None:
            status = "unresolved_source_entity"
            logger.warning(
                "Skipping relationship from %r because source entity was not found "
                "or could not be resolved.",
                relationship.source,
            )
        elif target_class_id is None:
            status = "unresolved_target_entity"
            logger.warning(
                "Skipping relationship to %r because target entity was not found "
                "or could not be resolved.",
                relationship.target,
            )
        elif source_class_name is None or target_class_name is None:
            status = "unresolved_class_name"
            logger.warning(
                "Skipping relationship %r-%r-%r because a final class name could "
                "not be determined.",
                relationship.source,
                relationship.type,
                relationship.target,
            )
        else:
            exact_triple = (source_class_name, normalized_type, target_class_name)
            if exact_triple not in exact_triple_details:
                exact_relationship_triples.append(exact_triple)
                exact_triple_details[exact_triple] = {
                    "descriptions": [],
                    "mention_count": 0,
                }
            if relationship.description:
                exact_triple_details[exact_triple]["descriptions"].append(
                    relationship.description
                )
            exact_triple_details[exact_triple]["mention_count"] += 1

        relationship_rows.append(
            {
                **asdict(relationship),
                "normalized_type": normalized_type,
                "consolidated_relationship_type": None,
                "source_resolved_canonical_class_id": source_class_id,
                "source_resolved_canonical_class_name": source_class_name,
                "target_resolved_canonical_class_id": target_class_id,
                "target_resolved_canonical_class_name": target_class_name,
                "resolution_status": status,
            }
        )

    return relationship_rows, exact_relationship_triples, exact_triple_details


def _consolidate_relationship_types(
    exact_relationship_triples: list[tuple[str, str, str]],
    exact_triple_details: dict[tuple[str, str, str], dict[str, Any]],
    threshold: float,
    embedding_model_name: str | None,
) -> tuple[
    set[tuple[str, str, str]],
    list[ConsolidationDecision],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], str],
]:
    grouped_triples: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for domain_class, rel_type, range_class in exact_relationship_triples:
        grouped_triples.setdefault((domain_class, range_class), []).append(
            (domain_class, rel_type, range_class)
        )

    consolidated_triples: set[tuple[str, str, str]] = set()
    decisions: list[ConsolidationDecision] = []
    provenance_accumulator: dict[str, dict[str, Any]] = {}
    canonical_by_exact_triple: dict[tuple[str, str, str], str] = {}

    for (domain_class, range_class), group_triples in grouped_triples.items():
        consolidated_types: list[_ConsolidatedRelationshipType] = []

        for exact_triple in group_triples:
            _, rel_type, _ = exact_triple
            details = exact_triple_details[exact_triple]
            mention_count = int(details.get("mention_count") or 1)
            descriptions = _unique_strings(details.get("descriptions", []))
            normalized_type = _normalize_relationship_type(rel_type)

            exact_match = _find_exact_relationship_type_match(
                normalized_type=normalized_type,
                consolidated_types=consolidated_types,
            )
            if exact_match is not None:
                exact_match.aliases = _append_unique(exact_match.aliases, rel_type)
                exact_match.mention_count += mention_count
                canonical_by_exact_triple[exact_triple] = exact_match.canonical_name
                consolidated_triples.add(
                    (domain_class, exact_match.canonical_name, range_class)
                )
                decisions.append(
                    _relationship_type_decision(
                        decision_type="exact_match",
                        raw_relationship_type=rel_type,
                        resulting_relationship_type=exact_match.canonical_name,
                        matched_relationship_type=exact_match.canonical_name,
                        domain_class=domain_class,
                        range_class=range_class,
                        similarity_score=None,
                        threshold=threshold,
                    )
                )
                continue

            embedding = _embed_relationship_type(
                rel_type=rel_type,
                descriptions=descriptions,
                embedding_model_name=embedding_model_name,
            )
            best_match, best_similarity = _find_best_relationship_type_match(
                embedding=embedding,
                consolidated_types=consolidated_types,
            )

            if best_match is not None and best_similarity is not None:
                if best_similarity >= threshold:
                    best_match.aliases = _append_unique(best_match.aliases, rel_type)
                    best_match.embedding = _merge_embedding(
                        old_embedding=best_match.embedding,
                        old_count=best_match.mention_count,
                        new_embedding=embedding,
                        new_count=mention_count,
                    )
                    best_match.mention_count += mention_count
                    canonical_by_exact_triple[exact_triple] = best_match.canonical_name
                    consolidated_triples.add(
                        (domain_class, best_match.canonical_name, range_class)
                    )
                    decisions.append(
                        _relationship_type_decision(
                            decision_type="embedding_merge",
                            raw_relationship_type=rel_type,
                            resulting_relationship_type=best_match.canonical_name,
                            matched_relationship_type=best_match.canonical_name,
                            domain_class=domain_class,
                            range_class=range_class,
                            similarity_score=best_similarity,
                            threshold=threshold,
                        )
                    )
                    continue

            # The first exact-deduplicated type encountered in a domain/range
            # group becomes canonical. Later synonyms in the same group merge
            # into that first name when exact or embedding similarity matches.
            new_type = _ConsolidatedRelationshipType(
                domain_class=domain_class,
                range_class=range_class,
                canonical_name=normalized_type,
                aliases=[rel_type],
                mention_count=mention_count,
                embedding=embedding,
            )
            consolidated_types.append(new_type)
            canonical_by_exact_triple[exact_triple] = new_type.canonical_name
            consolidated_triples.add(
                (domain_class, new_type.canonical_name, range_class)
            )
            decisions.append(
                _relationship_type_decision(
                    decision_type="new_relationship_type",
                    raw_relationship_type=rel_type,
                    resulting_relationship_type=new_type.canonical_name,
                    matched_relationship_type=(
                        best_match.canonical_name if best_match is not None else None
                    ),
                    domain_class=domain_class,
                    range_class=range_class,
                    similarity_score=best_similarity,
                    threshold=threshold,
                )
            )

        _accumulate_relationship_type_provenance(
            provenance_accumulator=provenance_accumulator,
            consolidated_types=consolidated_types,
            domain_class=domain_class,
            range_class=range_class,
        )

    return (
        consolidated_triples,
        decisions,
        _finalize_relationship_type_provenance(provenance_accumulator),
        canonical_by_exact_triple,
    )


def _find_exact_relationship_type_match(
    normalized_type: str,
    consolidated_types: list[_ConsolidatedRelationshipType],
) -> _ConsolidatedRelationshipType | None:
    for consolidated_type in consolidated_types:
        known_names = [
            consolidated_type.canonical_name,
            *consolidated_type.aliases,
        ]
        if normalized_type in {
            _normalize_relationship_type(known_name)
            for known_name in known_names
        }:
            return consolidated_type
    return None


def _find_best_relationship_type_match(
    embedding: list[float],
    consolidated_types: list[_ConsolidatedRelationshipType],
) -> tuple[_ConsolidatedRelationshipType | None, float | None]:
    best_match: _ConsolidatedRelationshipType | None = None
    best_similarity: float | None = None

    for consolidated_type in consolidated_types:
        similarity = cosine_similarity(embedding, consolidated_type.embedding)
        if best_similarity is None or similarity > best_similarity:
            best_match = consolidated_type
            best_similarity = similarity

    return best_match, best_similarity


def _embed_relationship_type(
    rel_type: str,
    descriptions: list[str],
    embedding_model_name: str | None,
) -> list[float]:
    text = _normalize_relationship_type(rel_type)
    if descriptions:
        text = f"{text}. {' '.join(descriptions)}"
    if embedding_model_name is None:
        model = initialize_embedding_model()
    else:
        model = initialize_embedding_model(embedding_model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def _merge_embedding(
    old_embedding: list[float],
    old_count: int,
    new_embedding: list[float],
    new_count: int,
) -> list[float]:
    if len(old_embedding) != len(new_embedding):
        raise ValueError("Relationship type embeddings must have the same dimensionality")

    old_weight = max(old_count, 1)
    new_weight = max(new_count, 1)
    merged = [
        ((old_value * old_weight) + (new_value * new_weight))
        / (old_weight + new_weight)
        for old_value, new_value in zip(old_embedding, new_embedding)
    ]
    return _renormalize(merged)


def _renormalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _relationship_type_decision(
    decision_type: str,
    raw_relationship_type: str,
    resulting_relationship_type: str,
    matched_relationship_type: str | None,
    domain_class: str,
    range_class: str,
    similarity_score: float | None,
    threshold: float,
) -> ConsolidationDecision:
    return ConsolidationDecision(
        decision_type=decision_type,
        subject_type="relationship_type",
        raw_relationship_type=raw_relationship_type,
        resulting_relationship_type=resulting_relationship_type,
        matched_relationship_type=matched_relationship_type,
        domain_class=domain_class,
        range_class=range_class,
        similarity_score=similarity_score,
        threshold=threshold,
    )


def _accumulate_relationship_type_provenance(
    provenance_accumulator: dict[str, dict[str, Any]],
    consolidated_types: list[_ConsolidatedRelationshipType],
    domain_class: str,
    range_class: str,
) -> None:
    for consolidated_type in consolidated_types:
        item = provenance_accumulator.setdefault(
            consolidated_type.canonical_name,
            {
                "aliases": [],
                "mention_count": 0,
                "groups": set(),
            },
        )
        for alias in consolidated_type.aliases:
            item["aliases"] = _append_unique(item["aliases"], alias)
        item["mention_count"] += consolidated_type.mention_count
        item["groups"].add((domain_class, range_class))


def _finalize_relationship_type_provenance(
    provenance_accumulator: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for canonical_name in sorted(provenance_accumulator):
        item = provenance_accumulator[canonical_name]
        finalized[canonical_name] = {
            "aliases": sorted(item["aliases"]),
            "mention_count": item["mention_count"],
            "groups": [
                {
                    "domain_class": domain_class,
                    "range_class": range_class,
                }
                for domain_class, range_class in sorted(item["groups"])
            ],
        }
    return finalized


def _add_consolidated_relationship_types(
    relationship_rows: list[dict[str, Any]],
    consolidated_type_by_exact_triple: dict[tuple[str, str, str], str],
) -> None:
    for row in relationship_rows:
        source_class = row.get("source_resolved_canonical_class_name")
        normalized_type = row.get("normalized_type")
        target_class = row.get("target_resolved_canonical_class_name")
        if not source_class or not normalized_type or not target_class:
            continue
        row["consolidated_relationship_type"] = consolidated_type_by_exact_triple.get(
            (source_class, normalized_type, target_class)
        )


def _class_provenance(state: PipelineState) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for consolidated_class in sorted(
        state.consolidated_classes.values(),
        key=lambda item: item.canonical_name,
    ):
        provenance[consolidated_class.canonical_name] = {
            "aliases": consolidated_class.aliases,
            "mention_count": consolidated_class.mention_count,
            "source_chunk_indices": consolidated_class.source_chunk_indices,
        }
    return provenance


def _write_records(
    destination: Path,
    stem: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> dict[str, Path]:
    json_path = destination / f"{stem}.json"
    csv_path = destination / f"{stem}.csv"
    _write_json(json_path, rows)
    _write_csv(csv_path, rows, fieldnames)
    return {"json": json_path, "csv": csv_path}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_cell(row.get(key))
                    for key in fieldnames
                }
            )


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _append_unique(items: list[Any], item: Any) -> list[Any]:
    if item in items:
        return list(items)
    return [*items, item]


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _resolve_final_class_id(
    source_class_id: str,
    direct_resolution: dict[str, str],
    final_classes: dict[str, Any],
    seen: set[str],
) -> str | None:
    if source_class_id in final_classes:
        return source_class_id
    if source_class_id in seen:
        logger.warning(
            "Cycle detected while resolving class id %r through %s.",
            source_class_id,
            sorted(seen),
        )
        return None

    seen.add(source_class_id)
    next_class_id = direct_resolution.get(source_class_id)
    if next_class_id is None:
        return None
    return _resolve_final_class_id(
        source_class_id=next_class_id,
        direct_resolution=direct_resolution,
        final_classes=final_classes,
        seen=seen,
    )


def _find_class_id_by_raw_name(
    raw_name: str,
    final_classes: dict[str, Any],
) -> str | None:
    matches = [
        class_id
        for class_id, consolidated_class in final_classes.items()
        if raw_name == consolidated_class.canonical_name
        or raw_name in consolidated_class.aliases
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Raw class name %r matches multiple final classes %s; not resolving.",
            raw_name,
            matches,
        )
    return None


def _add_resolution(
    resolution: dict[str, str],
    source_class_id: str,
    final_class_id: str,
) -> None:
    if not source_class_id:
        return
    existing = resolution.get(source_class_id)
    if existing is not None and existing != final_class_id:
        logger.warning(
            "Class id %r resolves to both %r and %r; keeping %r.",
            source_class_id,
            existing,
            final_class_id,
            existing,
        )
        return
    resolution[source_class_id] = final_class_id


def _class_name(state: PipelineState, class_id: str | None) -> str | None:
    if class_id is None:
        return None
    consolidated_class = state.consolidated_classes.get(class_id)
    if consolidated_class is None:
        return None
    return consolidated_class.canonical_name


def _normalize_relationship_type(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.upper()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", slug)
