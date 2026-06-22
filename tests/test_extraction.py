from __future__ import annotations

import json
import logging

import pytest

from conftest import FakeLLMClient
from extraction.chunk_llm_extractor import extract_schema_candidates_for_chunk
from models import PipelineState, RawEntity


def valid_payload() -> str:
    return json.dumps(
        {
            "classes": [
                {
                    "id": "person",
                    "name": "Person",
                    "description": "A person described by the chunk.",
                }
            ],
            "entities": [
                {
                    "id": "alice",
                    "name": "Alice",
                    "class_id": "person",
                    "description": "Alice appears in the chunk.",
                }
            ],
            "relationships": [
                {
                    "source": "alice",
                    "target": "acme",
                    "type": "WORKS_FOR",
                    "description": "Alice works for Acme.",
                }
            ],
        }
    )


def state_with_known_target() -> PipelineState:
    return PipelineState(
        raw_entities=[
            RawEntity(
                id="acme",
                name="Acme",
                class_id="organization",
                description="Known organization.",
                chunk_idx=0,
            )
        ]
    )


def test_valid_json_is_parsed_and_validated(no_sleep: object) -> None:
    client = FakeLLMClient([valid_payload()])

    result = extract_schema_candidates_for_chunk(
        chunk="Alice works for Acme.",
        chunk_idx=3,
        state=state_with_known_target(),
        client=client,
        model="fake-model",
        sleep=no_sleep,
    )

    assert [raw_class.id for raw_class in result.raw_classes] == ["person"]
    assert result.raw_classes[0].chunk_idx == 3
    assert [entity.id for entity in result.raw_entities] == ["alice"]
    assert [relationship.type for relationship in result.raw_relationships] == ["WORKS_FOR"]
    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_malformed_json_uses_validation_retry_path(no_sleep: object) -> None:
    client = FakeLLMClient(["{bad json", valid_payload()])

    result = extract_schema_candidates_for_chunk(
        chunk="Alice works for Acme.",
        chunk_idx=1,
        state=state_with_known_target(),
        client=client,
        model="fake-model",
        validation_retries=2,
        validation_backoff_seconds=0.05,
        sleep=no_sleep,
    )

    assert [raw_class.id for raw_class in result.raw_classes] == ["person"]
    assert no_sleep.calls == [0.05]
    assert len(client.completions.calls) == 2


def test_missing_required_entity_field_is_rejected_after_validation_retries(
    no_sleep: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_payload = json.dumps(
        {
            "classes": [],
            "entities": [{"id": "alice", "name": "Alice", "description": "No class id."}],
            "relationships": [],
        }
    )
    client = FakeLLMClient([invalid_payload])

    with caplog.at_level(logging.WARNING):
        result = extract_schema_candidates_for_chunk(
            chunk="Alice appears.",
            chunk_idx=4,
            state=PipelineState(),
            client=client,
            model="fake-model",
            validation_retries=1,
            sleep=no_sleep,
        )

    assert result.raw_classes == []
    assert "entities[0].class_id is required" in caplog.text


def test_unknown_relationship_source_is_rejected_not_silently_accepted(
    no_sleep: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_payload = json.dumps(
        {
            "classes": [],
            "entities": [],
            "relationships": [
                {
                    "source": "missing",
                    "target": "also_missing",
                    "type": "WORKS_FOR",
                    "description": "Invalid references.",
                }
            ],
        }
    )
    client = FakeLLMClient([invalid_payload])

    with caplog.at_level(logging.WARNING):
        result = extract_schema_candidates_for_chunk(
            chunk="Invalid relation.",
            chunk_idx=5,
            state=PipelineState(),
            client=client,
            model="fake-model",
            validation_retries=1,
            sleep=no_sleep,
        )

    assert result.raw_relationships == []
    assert "source references unknown entity id 'missing'" in caplog.text


def test_transient_api_errors_use_separate_exponential_backoff_path(
    no_sleep: object,
) -> None:
    client = FakeLLMClient([TimeoutError("connection timeout"), valid_payload()])

    result = extract_schema_candidates_for_chunk(
        chunk="Alice works for Acme.",
        chunk_idx=6,
        state=state_with_known_target(),
        client=client,
        model="fake-model",
        api_retries=2,
        api_backoff_seconds=0.25,
        validation_retries=1,
        sleep=no_sleep,
    )

    assert [entity.id for entity in result.raw_entities] == ["alice"]
    assert no_sleep.calls == [0.25]
    assert len(client.completions.calls) == 2


def test_exhausted_transient_api_retries_returns_empty_result(
    no_sleep: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeLLMClient(
        [
            TimeoutError("connection timeout"),
            TimeoutError("connection timeout again"),
        ]
    )

    with caplog.at_level(logging.ERROR):
        result = extract_schema_candidates_for_chunk(
            chunk="Alice works for Acme.",
            chunk_idx=7,
            state=state_with_known_target(),
            client=client,
            model="fake-model",
            api_retries=2,
            api_backoff_seconds=0.25,
            validation_retries=3,
            sleep=no_sleep,
        )

    assert result.raw_classes == []
    assert result.raw_entities == []
    assert result.raw_relationships == []
    assert no_sleep.calls == [0.25]
    assert "Transient API failure for chunk 7 after 2 retries" in caplog.text
