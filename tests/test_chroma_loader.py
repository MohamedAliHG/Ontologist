from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from loading import chroma_chunk_loader as loader


class FakeCollection:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        limit = kwargs["limit"]
        offset = kwargs["offset"]
        page = self.records[offset : offset + limit]
        return {
            "ids": [record["id"] for record in page],
            "documents": [record["document"] for record in page],
            "metadatas": [record["metadata"] for record in page],
        }


class FakeClient:
    def __init__(self, collections: dict[str, FakeCollection]) -> None:
        self.collections = collections

    def list_collections(self) -> list[Any]:
        return [SimpleNamespace(name=name) for name in self.collections]

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


def install_fake_chromadb(
    monkeypatch: pytest.MonkeyPatch,
    client: FakeClient,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda path: client),
    )


def make_chroma_dir(tmp_path: Any) -> str:
    db_dir = tmp_path / "chroma"
    db_dir.mkdir()
    (db_dir / "chroma.sqlite3").write_text("", encoding="utf-8")
    return str(db_dir)


def test_loads_chunks_ordered_by_consistent_metadata_field(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        [
            {"id": "b", "document": "second", "metadata": {"namespace": "ns", "chunk_index": 1}},
            {"id": "a", "document": "first", "metadata": {"namespace": "ns", "chunk_index": 0}},
            {"id": "c", "document": "third", "metadata": {"namespace": "ns", "chunk_index": "2"}},
        ]
    )
    install_fake_chromadb(monkeypatch, FakeClient({"docs": collection}))

    chunks = loader.load_chunks_from_chromadb(
        make_chroma_dir(tmp_path),
        collection_name="docs",
        namespace="ns",
        batch_size=2,
    )

    assert chunks == ["first", "second", "third"]
    assert [call["offset"] for call in collection.calls] == [0, 2]


def test_auto_detection_uses_actual_priority_order(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        [
            {
                "id": "one",
                "document": "by chunk_idx first",
                "metadata": {"namespace": "ns", "chunk_idx": 0, "page_number": 99},
            },
            {
                "id": "two",
                "document": "by chunk_idx second",
                "metadata": {"namespace": "ns", "chunk_idx": 1, "page_number": 1},
            },
        ]
    )
    install_fake_chromadb(monkeypatch, FakeClient({"docs": collection}))

    loaded = loader.load_chunks_from_chromadb_with_info(
        make_chroma_dir(tmp_path),
        collection_name="docs",
        namespace="ns",
    )

    assert loaded.ordering_field == "chunk_idx"
    assert loaded.chunks == ["by chunk_idx first", "by chunk_idx second"]


def test_no_usable_order_field_raises_and_lists_found_metadata_keys(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        [
            {"id": "a", "document": "alpha", "metadata": {"namespace": "ns", "section": "A"}},
            {"id": "b", "document": "beta", "metadata": {"namespace": "ns", "page_label": "B"}},
        ]
    )
    install_fake_chromadb(monkeypatch, FakeClient({"docs": collection}))

    with pytest.raises(loader.ChromaChunkOrderingError) as exc_info:
        loader.load_chunks_from_chromadb(
            make_chroma_dir(tmp_path),
            collection_name="docs",
            namespace="ns",
        )

    message = str(exc_info.value)
    assert "Chunk ordering could not be determined" in message
    assert "namespace, page_label, section" in message


def test_explicit_order_field_override_bypasses_auto_detection(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(
        [
            {
                "id": "b",
                "document": "second",
                "metadata": {"namespace": "ns", "chunk_index": "not-an-int", "custom_order": 2},
            },
            {
                "id": "a",
                "document": "first",
                "metadata": {"namespace": "ns", "chunk_index": "not-an-int", "custom_order": 1},
            },
        ]
    )
    install_fake_chromadb(monkeypatch, FakeClient({"docs": collection}))

    loaded = loader.load_chunks_from_chromadb_with_info(
        make_chroma_dir(tmp_path),
        collection_name="docs",
        namespace="ns",
        order_field="custom_order",
    )

    assert loaded.ordering_field == "custom_order"
    assert loaded.chunks == ["first", "second"]


def test_collection_not_found_raises_distinct_exception(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_chromadb(monkeypatch, FakeClient({"other": FakeCollection([])}))

    with pytest.raises(loader.ChromaCollectionNotFoundError) as exc_info:
        loader.load_chunks_from_chromadb(
            make_chroma_dir(tmp_path),
            collection_name="missing",
            namespace="ns",
        )

    assert "Available collections: other" in str(exc_info.value)


def test_empty_namespace_raises_distinct_exception(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_chromadb(monkeypatch, FakeClient({"docs": FakeCollection([])}))

    with pytest.raises(loader.ChromaNamespaceNotFoundError) as exc_info:
        loader.load_chunks_from_chromadb(
            make_chroma_dir(tmp_path),
            collection_name="docs",
            namespace="missing",
        )

    assert "No records found" in str(exc_info.value)
    assert "namespace" in str(exc_info.value)
