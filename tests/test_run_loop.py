from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from extraction.chunk_llm_extractor import ChunkExtractionResult
from models import ConsolidatedClass, ConsolidationDecision, PipelineState, RawClass, RawEntity
from pipeline import pass1_runner


def test_process_chunks_updates_state_in_chunk_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at_extraction: list[tuple[str, int, int, tuple[str, ...]]] = []

    def fake_extract(chunk: str, chunk_idx: int, state: PipelineState, **kwargs: Any) -> ChunkExtractionResult:
        observed_at_extraction.append(
            (
                chunk,
                chunk_idx,
                len(state.raw_classes),
                tuple(sorted(state.consolidated_classes)),
            )
        )
        if chunk == "failed-but-skipped":
            return ChunkExtractionResult()
        raw = RawClass(
            id=f"class_{chunk_idx}",
            name=f"Class {chunk_idx}",
            description=f"Class from chunk {chunk_idx}.",
            chunk_idx=chunk_idx,
        )
        return ChunkExtractionResult(
            raw_classes=[raw],
            raw_entities=[
                RawEntity(
                    id=f"entity_{chunk_idx}",
                    name=f"Entity {chunk_idx}",
                    class_id=raw.id,
                    description="Entity from chunk.",
                    chunk_idx=chunk_idx,
                )
            ],
        )

    def fake_consolidate(
        new_raw_classes: list[RawClass],
        consolidated_classes: dict[str, ConsolidatedClass],
        threshold: float,
        embedding_model_name: str,
    ) -> tuple[dict[str, ConsolidatedClass], list[ConsolidationDecision]]:
        updated = dict(consolidated_classes)
        decisions: list[ConsolidationDecision] = []
        for raw in new_raw_classes:
            updated[raw.id] = ConsolidatedClass(
                id=raw.id,
                canonical_name=raw.name,
                description=raw.description,
                aliases=[raw.name],
                mention_count=1,
                embedding=[1.0, 0.0],
                source_chunk_indices=[raw.chunk_idx],
            )
            decisions.append(
                ConsolidationDecision(
                    decision_type="new_class",
                    raw_class=raw,
                    resulting_class_id=raw.id,
                    threshold=threshold,
                )
            )
        return updated, decisions

    monkeypatch.setattr(pass1_runner, "extract_schema_candidates_for_chunk", fake_extract)
    monkeypatch.setattr(pass1_runner, "consolidate_raw_classes", fake_consolidate)

    state = pass1_runner.process_chunks(
        chunks=["first", "failed-but-skipped", "third"],
        threshold=0.82,
        extraction_kwargs={"model": "fake"},
        embedding_model_name="fake-embedding",
    )

    assert observed_at_extraction == [
        ("first", 0, 0, ()),
        ("failed-but-skipped", 1, 1, ("class_0",)),
        ("third", 2, 1, ("class_0",)),
    ]
    assert [raw.chunk_idx for raw in state.raw_classes] == [0, 2]
    assert [entity.id for entity in state.raw_entities] == ["entity_0", "entity_2"]
    assert sorted(state.consolidated_classes) == ["class_0", "class_2"]
    assert len(state.chunk_log) == 3
    assert state.chunk_log[1]["raw_classes_extracted"] == 0
    assert state.chunk_log[1]["consolidation_decisions"] == {
        "exact_match": 0,
        "embedding_merge": 0,
        "new_class": 0,
    }


def test_main_uses_chroma_loader_and_writes_export_after_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loader_calls: list[dict[str, Any]] = []
    processed_chunks: list[list[str]] = []

    def fake_load_chunks_from_chromadb(**kwargs: Any) -> list[str]:
        loader_calls.append(kwargs)
        return ["chunk-a", "chunk-b"]

    def fake_process_chunks(
        chunks: list[str],
        threshold: float,
        extraction_kwargs: dict[str, Any],
        embedding_model_name: str,
    ) -> PipelineState:
        processed_chunks.append(chunks)
        assert threshold == 0.7
        assert extraction_kwargs["client"] == "fake-client"
        assert extraction_kwargs["model"] == "fake-model"
        assert embedding_model_name == "fake-embedding"
        return PipelineState()

    def fake_export_schema_profile(**kwargs: Any) -> Any:
        assert kwargs["output_dir"] == str(tmp_path)
        assert kwargs["relationship_type_threshold"] == 0.7
        assert kwargs["relationship_type_embedding_model_name"] == "fake-embedding"
        return SimpleNamespace(
            document_id="doc",
            output_dir=tmp_path,
            schema_path=tmp_path / "schema_profile.json",
            counts={},
        )

    import loading.chroma_chunk_loader as chroma_loader

    monkeypatch.setattr(chroma_loader, "load_chunks_from_chromadb", fake_load_chunks_from_chromadb)
    monkeypatch.setattr(pass1_runner, "make_llm_client", lambda **kwargs: "fake-client")
    monkeypatch.setattr(pass1_runner, "process_chunks", fake_process_chunks)
    monkeypatch.setattr(pass1_runner, "export_schema_profile", fake_export_schema_profile)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pass1.py",
            "db-path",
            "collection",
            "namespace",
            str(tmp_path),
            "--order-field",
            "chunk_index",
            "--batch-size",
            "25",
            "--threshold",
            "0.7",
            "--model",
            "fake-model",
            "--embedding-model",
            "fake-embedding",
        ],
    )

    pass1_runner.main()

    assert loader_calls == [
        {
            "chromadb_path": "db-path",
            "collection_name": "collection",
            "namespace": "namespace",
            "order_field": "chunk_index",
            "batch_size": 25,
        }
    ]
    assert processed_chunks == [["chunk-a", "chunk-b"]]
