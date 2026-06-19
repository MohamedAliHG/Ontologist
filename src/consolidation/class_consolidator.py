from __future__ import annotations

import math
import os
import re
import string
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from config.settings import DEFAULT_EMBEDDING_MODEL_NAME

if TYPE_CHECKING:
    from models import ConsolidatedClass, ConsolidationDecision, RawClass


_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_MODEL_NAME: str | None = None


def initialize_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> Any:
    """Load the sentence-transformers model once for pipeline startup.

    Calling this at startup satisfies the "load once and reuse" path. If callers
    skip explicit startup initialization, consolidation lazily initializes the
    same singleton on first use and reuses it after that.
    """

    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_NAME

    if (
        _EMBEDDING_MODEL is not None
        and _EMBEDDING_MODEL_NAME == model_name
    ):
        return _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME != model_name:
        raise RuntimeError(
            "A sentence-transformers model is already loaded as "
            f"{_EMBEDDING_MODEL_NAME!r}. Refusing to load {model_name!r} because "
            "the consolidator must reuse one model instance across the run."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' package is required for class "
            "consolidation. Install it before calling consolidate_raw_classes()."
        ) from exc

    _EMBEDDING_MODEL = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_NAME = model_name
    return _EMBEDDING_MODEL


def consolidate_raw_classes(
    new_raw_classes: list["RawClass"],
    consolidated_classes: dict[str, "ConsolidatedClass"],
    threshold: float,
) -> tuple[dict[str, "ConsolidatedClass"], list["ConsolidationDecision"]]:
    """Merge newly proposed raw classes into consolidated classes.

    Args:
        new_raw_classes: RawClass proposals from one chunk extraction.
        consolidated_classes: Current canonical classes keyed by id.
        threshold: Minimum cosine similarity for embedding-based merge.

    Returns:
        A new consolidated_classes dict plus new ConsolidationDecision entries.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    updated = dict(consolidated_classes)
    decisions: list[ConsolidationDecision] = []

    for raw_class in new_raw_classes:
        exact_match_id = _find_exact_match(raw_class, updated)
        if exact_match_id is not None:
            matched = updated[exact_match_id]
            updated[exact_match_id] = _merge_exact_match(matched, raw_class)
            decisions.append(
                _make_decision(
                    raw_class=raw_class,
                    decision_type="exact_match",
                    matched_class=matched,
                    resulting_class_id=matched.id,
                    similarity_score=None,
                    threshold=threshold,
                )
            )
            continue

        new_embedding = _embed_raw_class(raw_class)
        best_class_id, best_similarity = _find_best_embedding_match(
            new_embedding,
            updated,
        )

        if best_class_id is not None and best_similarity is not None:
            matched = updated[best_class_id]
            if best_similarity >= threshold:
                updated[best_class_id] = _merge_embedding_match(
                    consolidated_class=matched,
                    raw_class=raw_class,
                    new_embedding=new_embedding,
                )
                decisions.append(
                    _make_decision(
                        raw_class=raw_class,
                        decision_type="embedding_merge",
                        matched_class=matched,
                        resulting_class_id=matched.id,
                        similarity_score=best_similarity,
                        threshold=threshold,
                    )
                )
                continue

        new_class = _new_consolidated_class(
            raw_class=raw_class,
            embedding=new_embedding,
            existing_ids=set(updated),
        )
        updated[new_class.id] = new_class
        decisions.append(
            _make_decision(
                raw_class=raw_class,
                decision_type="new_class",
                matched_class=(
                    updated[best_class_id]
                    if best_class_id is not None and best_class_id in updated
                    else None
                ),
                resulting_class_id=new_class.id,
                similarity_score=best_similarity,
                threshold=threshold,
            )
        )

    return updated, decisions


def slugify(name: str) -> str:
    """Return a lowercase snake_case id with non-alphanumerics stripped."""

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "class"


def normalize_class_name(name: str) -> str:
    """Normalize a class name for exact canonical-name/alias matching."""

    cleaned = name.lower().strip()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _find_exact_match(
    raw_class: "RawClass",
    consolidated_classes: dict[str, "ConsolidatedClass"],
) -> str | None:
    raw_normalized = normalize_class_name(raw_class.name)
    for class_id, consolidated_class in consolidated_classes.items():
        names = [
            consolidated_class.canonical_name,
            *consolidated_class.aliases,
        ]
        if raw_normalized in {normalize_class_name(name) for name in names}:
            return class_id
    return None


def _merge_exact_match(
    consolidated_class: "ConsolidatedClass",
    raw_class: "RawClass",
) -> "ConsolidatedClass":
    return replace(
        consolidated_class,
        aliases=_append_unique(consolidated_class.aliases, raw_class.name),
        mention_count=consolidated_class.mention_count + 1,
        source_chunk_indices=[
            *consolidated_class.source_chunk_indices,
            raw_class.chunk_idx,
        ],
    )


def _merge_embedding_match(
    consolidated_class: "ConsolidatedClass",
    raw_class: "RawClass",
    new_embedding: list[float],
) -> "ConsolidatedClass":
    old_count = max(consolidated_class.mention_count, 1)
    old_embedding = consolidated_class.embedding
    if old_embedding is None:
        merged_embedding = new_embedding
    else:
        if len(old_embedding) != len(new_embedding):
            raise ValueError("Embeddings must have the same dimensionality")
        merged_embedding = _renormalize(
            [
                ((old_value * old_count) + new_value) / (old_count + 1)
                for old_value, new_value in zip(old_embedding, new_embedding)
            ]
        )

    return replace(
        consolidated_class,
        aliases=_append_unique(consolidated_class.aliases, raw_class.name),
        mention_count=consolidated_class.mention_count + 1,
        embedding=merged_embedding,
        source_chunk_indices=[
            *consolidated_class.source_chunk_indices,
            raw_class.chunk_idx,
        ],
    )


def _new_consolidated_class(
    raw_class: "RawClass",
    embedding: list[float],
    existing_ids: set[str],
) -> "ConsolidatedClass":
    ConsolidatedClass = _load_model_classes()[1]
    base_id = slugify(raw_class.name)
    class_id = _unique_id(base_id, existing_ids)
    return ConsolidatedClass(
        id=class_id,
        canonical_name=raw_class.name,
        description=raw_class.description,
        aliases=[raw_class.name],
        mention_count=1,
        embedding=embedding,
        source_chunk_indices=[raw_class.chunk_idx],
    )


def _find_best_embedding_match(
    new_embedding: list[float],
    consolidated_classes: dict[str, "ConsolidatedClass"],
) -> tuple[str | None, float | None]:
    best_class_id: str | None = None
    best_similarity: float | None = None

    for class_id, consolidated_class in consolidated_classes.items():
        if consolidated_class.embedding is None:
            continue
        similarity = cosine_similarity(new_embedding, consolidated_class.embedding)
        if best_similarity is None or similarity > best_similarity:
            best_class_id = class_id
            best_similarity = similarity

    return best_class_id, best_similarity


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""

    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def _embed_raw_class(raw_class: "RawClass") -> list[float]:
    text = f"{raw_class.name}. {raw_class.description}"
    model_name = os.getenv("SCHEMA_CLASS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL_NAME)
    model = initialize_embedding_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def _renormalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _append_unique(items: list[Any], item: Any) -> list[Any]:
    if item in items:
        return list(items)
    return [*items, item]


def _unique_id(base_id: str, existing_ids: set[str]) -> str:
    if base_id not in existing_ids:
        return base_id
    suffix = 2
    while f"{base_id}_{suffix}" in existing_ids:
        suffix += 1
    return f"{base_id}_{suffix}"


def _make_decision(
    raw_class: "RawClass",
    decision_type: str,
    matched_class: "ConsolidatedClass | None",
    resulting_class_id: str,
    similarity_score: float | None,
    threshold: float,
) -> "ConsolidationDecision":
    ConsolidationDecision = _load_model_classes()[2]
    return ConsolidationDecision(
        raw_class=raw_class,
        decision_type=decision_type,
        resulting_class_id=resulting_class_id,
        matched_class_id=matched_class.id if matched_class is not None else None,
        matched_class_name=(
            matched_class.canonical_name if matched_class is not None else None
        ),
        similarity_score=similarity_score,
        threshold=threshold,
    )


def _load_model_classes() -> tuple[Any, Any, Any]:
    from models import ConsolidatedClass, ConsolidationDecision, RawClass

    return RawClass, ConsolidatedClass, ConsolidationDecision
