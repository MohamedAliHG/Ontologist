from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from config.settings import (
    DEFAULT_API_RETRIES,
    DEFAULT_GROQ_MODEL,
    DEFAULT_VALIDATION_RETRIES,
)
from prompts.schema_candidate_extraction import (
    SYSTEM_PROMPT,
    build_schema_candidate_prompt,
)

if TYPE_CHECKING:
    from models import RawClass, RawEntity, RawRelationship


logger = logging.getLogger(__name__)

CLASS_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
RELATIONSHIP_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class ChunkExtractionResult:
    """Raw records from one chunk, ready for the caller to append."""

    raw_classes: list[RawClass] = field(default_factory=list)
    raw_entities: list[RawEntity] = field(default_factory=list)
    raw_relationships: list[RawRelationship] = field(default_factory=list)


class ExtractionValidationError(ValueError):
    """Raised when the LLM response is not valid candidate JSON."""


def extract_schema_candidates_for_chunk(
    chunk: str,
    chunk_idx: int,
    state: Any,
    *,
    model: str | None = None,
    provider: str = "groq",
    base_url: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    api_retries: int = DEFAULT_API_RETRIES,
    validation_backoff_seconds: float = 0.5,
    api_backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ChunkExtractionResult:
    """Extract raw candidates from one chunk without mutating pipeline state.

    Args:
        chunk: Text chunk to analyze.
        chunk_idx: Index of this chunk in the source chunk list.
        state: Current PipelineState, or any object with the same relevant attrs.
        model: Chat model name. Defaults to GROQ_MODEL or DEFAULT_GROQ_MODEL.
        provider: "groq" or "openai". Use "openai" for OpenAI-compatible local
            endpoints when passing base_url.
        base_url: Optional OpenAI-compatible endpoint base URL.
        api_key: Optional API key. If omitted, provider-specific env vars are used.
        client: Optional prebuilt/fake client for tests.
        validation_retries: Number of JSON/validation attempts.
        api_retries: Number of transient API attempts per validation attempt.
        validation_backoff_seconds: Backoff between validation retries.
        api_backoff_seconds: Base exponential backoff for transient API errors.
        sleep: Sleep function, injectable for tests.

    Returns:
        ChunkExtractionResult containing the three raw lists to append.
    """

    if not chunk.strip():
        logger.warning("Chunk %s is empty; returning no extraction candidates.", chunk_idx)
        return ChunkExtractionResult()
    if validation_retries <= 0:
        raise ValueError("validation_retries must be a positive integer")
    if api_retries <= 0:
        raise ValueError("api_retries must be a positive integer")

    prompt = _build_user_prompt(chunk=chunk, state=state)
    selected_model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    llm_client = client or _make_client(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
    )

    for validation_attempt in range(1, validation_retries + 1):
        try:
            response_text = _call_llm_with_api_retries(
                client=llm_client,
                model=selected_model,
                prompt=prompt,
                api_retries=api_retries,
                api_backoff_seconds=api_backoff_seconds,
                sleep=sleep,
            )
            payload = _parse_and_validate_response(response_text, state=state)
            return _payload_to_result(payload=payload, chunk_idx=chunk_idx)
        except ExtractionValidationError as exc:
            logger.warning(
                "Validation failure for chunk %s on attempt %s/%s: %s",
                chunk_idx,
                validation_attempt,
                validation_retries,
                exc,
            )
            if validation_attempt < validation_retries:
                sleep(validation_backoff_seconds)
        except Exception as exc:
            if _is_transient_api_error(exc):
                logger.error(
                    "Transient API failure for chunk %s after %s retries: %s",
                    chunk_idx,
                    api_retries,
                    exc,
                )
            else:
                logger.error(
                    "Non-retryable LLM failure for chunk %s: %s",
                    chunk_idx,
                    exc,
                )
            return ChunkExtractionResult()

    logger.error(
        "All validation retries failed for chunk %s; returning no candidates.",
        chunk_idx,
    )
    return ChunkExtractionResult()


def _make_client(
    provider: str,
    base_url: str | None,
    api_key: str | None,
) -> Any:
    provider_normalized = provider.lower()
    if provider_normalized == "groq" and base_url is None:
        return _make_groq_client(api_key=api_key)
    if provider_normalized in {"openai", "local", "groq"}:
        return _make_openai_compatible_client(base_url=base_url, api_key=api_key)
    raise ValueError("provider must be 'groq', 'openai', or 'local'")


def _make_groq_client(api_key: str | None) -> Any:
    try:
        from groq import Groq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'groq' package is required for provider='groq'. Install it or "
            "pass provider='openai' with an OpenAI-compatible client/base_url."
        ) from exc

    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when calling the Groq client. Set it in the "
            "environment or pass api_key=..."
        )
    return Groq(api_key=resolved_api_key)


def _make_openai_compatible_client(
    base_url: str | None,
    api_key: str | None,
) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'openai' package is required for OpenAI-compatible local "
            "endpoints. Install it or pass a prebuilt client=... for tests."
        ) from exc

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "local"
    kwargs: dict[str, str] = {"api_key": resolved_api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _call_llm_with_api_retries(
    client: Any,
    model: str,
    prompt: str,
    api_retries: int,
    api_backoff_seconds: float,
    sleep: Callable[[float], None],
) -> str:
    for attempt in range(1, api_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ExtractionValidationError("LLM returned empty content.")
            return content
        except ExtractionValidationError:
            raise
        except Exception as exc:
            if not _is_transient_api_error(exc) or attempt == api_retries:
                raise
            delay = api_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Transient API error on attempt %s/%s: %s. Retrying in %.2fs.",
                attempt,
                api_retries,
                exc,
                delay,
            )
            sleep(delay)

    raise RuntimeError("Unreachable API retry state.")


def _build_user_prompt(chunk: str, state: Any) -> str:
    prompt_payload = {
        "current_consolidated_classes": _compact_consolidated_classes(state),
        "running_raw_entities": _compact_raw_entities(state),
        "running_raw_relationships": _compact_raw_relationships(state),
        "chunk_text": chunk,
    }
    return build_schema_candidate_prompt(prompt_payload)


def _compact_consolidated_classes(state: Any) -> list[dict[str, str]]:
    classes = getattr(state, "consolidated_classes", {}) or {}
    compact: list[dict[str, str]] = []
    for class_id in sorted(classes):
        item = classes[class_id]
        compact.append(
            {
                "id": str(getattr(item, "id")),
                "canonical_name": str(getattr(item, "canonical_name")),
                "description": str(getattr(item, "description")),
            }
        )
    return compact


def _compact_raw_entities(state: Any) -> list[dict[str, str]]:
    seen: set[str] = set()
    compact: list[dict[str, str]] = []
    for entity in getattr(state, "raw_entities", []) or []:
        entity_id = str(getattr(entity, "id"))
        if entity_id in seen:
            continue
        seen.add(entity_id)
        compact.append(
            {
                "id": entity_id,
                "name": str(getattr(entity, "name")),
                "class_id": str(getattr(entity, "class_id")),
            }
        )
    return compact


def _compact_raw_relationships(state: Any) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    compact: list[dict[str, str]] = []
    for relationship in getattr(state, "raw_relationships", []) or []:
        source = str(getattr(relationship, "source"))
        rel_type = str(getattr(relationship, "type"))
        target = str(getattr(relationship, "target"))
        key = (source, rel_type, target)
        if key in seen:
            continue
        seen.add(key)
        compact.append({"source": source, "type": rel_type, "target": target})
    return compact


def _parse_and_validate_response(response_text: str, state: Any) -> dict[str, Any]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ExtractionValidationError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ExtractionValidationError("Response JSON must be an object.")

    for top_level_key in ("classes", "entities", "relationships"):
        if top_level_key not in payload:
            raise ExtractionValidationError(
                f"Response is missing top-level key {top_level_key!r}."
            )
        if not isinstance(payload[top_level_key], list):
            raise ExtractionValidationError(
                f"Response field {top_level_key!r} must be a list."
            )

    _validate_classes(payload["classes"])
    _validate_entities(payload["entities"])
    _validate_relationships(payload["relationships"], payload["entities"], state)
    return payload


def _validate_classes(classes: list[Any]) -> None:
    for idx, item in enumerate(classes):
        _require_object(item, f"classes[{idx}]")
        _require_non_empty_string(item, "id", f"classes[{idx}]")
        _require_non_empty_string(item, "name", f"classes[{idx}]")
        _require_non_empty_string(item, "description", f"classes[{idx}]")
        if not CLASS_ID_PATTERN.fullmatch(item["id"]):
            raise ExtractionValidationError(
                f"classes[{idx}].id must be a lowercase snake_case slug."
            )


def _validate_entities(entities: list[Any]) -> None:
    for idx, item in enumerate(entities):
        _require_object(item, f"entities[{idx}]")
        _require_non_empty_string(item, "id", f"entities[{idx}]")
        _require_non_empty_string(item, "name", f"entities[{idx}]")
        _require_non_empty_string(item, "class_id", f"entities[{idx}]")
        _require_non_empty_string(item, "description", f"entities[{idx}]")


def _validate_relationships(
    relationships: list[Any],
    new_entities: list[Any],
    state: Any,
) -> None:
    known_entity_ids = _existing_entity_ids(state)
    known_entity_ids.update(item["id"] for item in new_entities)

    for idx, item in enumerate(relationships):
        _require_object(item, f"relationships[{idx}]")
        _require_non_empty_string(item, "source", f"relationships[{idx}]")
        _require_non_empty_string(item, "target", f"relationships[{idx}]")
        _require_non_empty_string(item, "type", f"relationships[{idx}]")
        _require_non_empty_string(item, "description", f"relationships[{idx}]")
        if not RELATIONSHIP_TYPE_PATTERN.fullmatch(item["type"]):
            raise ExtractionValidationError(
                f"relationships[{idx}].type must be UPPER_SNAKE_CASE."
            )
        if item["source"] not in known_entity_ids:
            raise ExtractionValidationError(
                f"relationships[{idx}].source references unknown entity id "
                f"{item['source']!r}."
            )
        if item["target"] not in known_entity_ids:
            raise ExtractionValidationError(
                f"relationships[{idx}].target references unknown entity id "
                f"{item['target']!r}."
            )


def _require_object(item: Any, location: str) -> None:
    if not isinstance(item, dict):
        raise ExtractionValidationError(f"{location} must be an object.")


def _require_non_empty_string(item: dict[str, Any], key: str, location: str) -> None:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExtractionValidationError(
            f"{location}.{key} is required and must be a non-empty string."
        )


def _existing_entity_ids(state: Any) -> set[str]:
    return {
        str(getattr(entity, "id"))
        for entity in getattr(state, "raw_entities", []) or []
        if getattr(entity, "id", None) is not None
    }


def _payload_to_result(payload: dict[str, Any], chunk_idx: int) -> ChunkExtractionResult:
    RawClass, RawEntity, RawRelationship = _load_model_classes()
    return ChunkExtractionResult(
        raw_classes=[
            RawClass(
                id=item["id"].strip(),
                name=item["name"].strip(),
                description=item["description"].strip(),
                chunk_idx=chunk_idx,
            )
            for item in payload["classes"]
        ],
        raw_entities=[
            RawEntity(
                id=item["id"].strip(),
                name=item["name"].strip(),
                class_id=item["class_id"].strip(),
                description=item["description"].strip(),
                chunk_idx=chunk_idx,
            )
            for item in payload["entities"]
        ],
        raw_relationships=[
            RawRelationship(
                source=item["source"].strip(),
                target=item["target"].strip(),
                type=item["type"].strip(),
                description=item["description"].strip(),
                chunk_idx=chunk_idx,
            )
            for item in payload["relationships"]
        ],
    )


def _load_model_classes() -> tuple[Any, Any, Any]:
    from models import RawClass, RawEntity, RawRelationship

    return RawClass, RawEntity, RawRelationship


def _is_transient_api_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    transient_markers = (
        "ratelimit",
        "rate_limit",
        "timeout",
        "connection",
        "temporar",
        "serviceunavailable",
        "service unavailable",
        "internalserver",
        "internal server",
        "too many requests",
        "503",
        "502",
        "504",
    )
    return any(marker in name or marker in message for marker in transient_markers)
