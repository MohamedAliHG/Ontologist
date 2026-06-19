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

from models import PipelineState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaExportResult:
    """Paths and counts produced by the schema profile export stage."""

    output_dir: Path
    schema_path: Path
    artifact_paths: dict[str, dict[str, Path]]
    counts: dict[str, int]
    document_id: str


def export_schema_profile(
    state: PipelineState,
    output_dir: str | Path,
    document_id: str | None = None,
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
    relationship_rows, strict_relationships = _build_relationship_rows_and_triples(
        state=state,
        entity_lookup=entity_lookup,
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
        "consolidation_log": [asdict(entry) for entry in state.consolidation_log],
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
    }

    return SchemaExportResult(
        output_dir=destination,
        schema_path=schema_path,
        artifact_paths=artifact_paths,
        counts=counts,
        document_id=resolved_document_id,
    )


def _build_class_resolution_map(state: PipelineState) -> dict[str, str]:
    resolution: dict[str, str] = {}
    final_classes = state.consolidated_classes

    for class_id in final_classes:
        resolution[class_id] = class_id

    for entry in state.consolidation_log:
        raw_class = entry.raw_class
        resulting_class_id = _valid_final_class_id(
            getattr(entry, "resulting_class_id", None),
            final_classes,
        )
        if resulting_class_id is None:
            resulting_class_id = _valid_final_class_id(
                getattr(entry, "matched_class_id", None),
                final_classes,
            )
        if resulting_class_id is None:
            resulting_class_id = _find_class_id_by_raw_name(raw_class.name, final_classes)

        if resulting_class_id is None:
            logger.warning(
                "Could not resolve raw class %r (%r) from consolidation log.",
                raw_class.id,
                raw_class.name,
            )
            continue

        _add_resolution(resolution, raw_class.id, resulting_class_id)
        _add_resolution(resolution, _slugify(raw_class.name), resulting_class_id)

    slug_candidates: dict[str, set[str]] = {}
    for class_id, consolidated_class in final_classes.items():
        for name in [consolidated_class.canonical_name, *consolidated_class.aliases]:
            slug_candidates.setdefault(_slugify(name), set()).add(class_id)

    for slug, class_ids in slug_candidates.items():
        if len(class_ids) == 1:
            _add_resolution(resolution, slug, next(iter(class_ids)))
        else:
            logger.warning(
                "Class id slug %r is ambiguous across final classes %s; not using it.",
                slug,
                sorted(class_ids),
            )

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
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    relationship_rows: list[dict[str, Any]] = []
    strict_relationships: set[tuple[str, str, str]] = set()

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
            strict_relationships.add(
                (source_class_name, normalized_type, target_class_name)
            )

        relationship_rows.append(
            {
                **asdict(relationship),
                "normalized_type": normalized_type,
                "source_resolved_canonical_class_id": source_class_id,
                "source_resolved_canonical_class_name": source_class_name,
                "target_resolved_canonical_class_id": target_class_id,
                "target_resolved_canonical_class_name": target_class_name,
                "resolution_status": status,
            }
        )

    return relationship_rows, strict_relationships


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


def _valid_final_class_id(
    class_id: str | None,
    final_classes: dict[str, Any],
) -> str | None:
    if class_id is None:
        return None
    return class_id if class_id in final_classes else None


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
