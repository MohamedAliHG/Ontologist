from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ORDER_FIELDS = ("chunk_index", "chunk_idx", "page_number", "order")
DEFAULT_BATCH_SIZE = 1_000
NAMESPACE_METADATA_FIELD = "namespace"

logger = logging.getLogger(__name__)


class ChromaChunkLoaderError(Exception):
    """Base class for Chroma chunk loader errors."""


class ChromaPathError(ChromaChunkLoaderError):
    """Raised when the ChromaDB path is missing or not a persistent DB directory."""


class ChromaCollectionNotFoundError(ChromaChunkLoaderError):
    """Raised when the requested collection does not exist."""


class ChromaNamespaceNotFoundError(ChromaChunkLoaderError):
    """Raised when no chunks exist for the requested namespace."""


class ChromaChunkOrderingError(ChromaChunkLoaderError):
    """Raised when deterministic chunk ordering cannot be resolved."""


class ChromaChunkContentError(ChromaChunkLoaderError):
    """Raised when retrieved Chroma records do not contain usable string chunks."""


class ChromaDependencyError(ChromaChunkLoaderError):
    """Raised when the chromadb package is not installed."""


@dataclass(frozen=True)
class LoadedChunks:
    """Chunks plus load metadata for callers that want visibility without logs."""

    chunks: list[str]
    count: int
    ordering_field: str


def load_chunks_from_chromadb(
    chromadb_path: str,
    collection_name: str,
    namespace: str,
    order_field: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    """Return all chunk texts for a collection namespace in deterministic order.

    Args:
        chromadb_path: Path to the local persistent ChromaDB directory.
        collection_name: Name of the ChromaDB collection to read.
        namespace: Metadata namespace value used to select records.
        order_field: Optional metadata field to sort by. If omitted, common order
            fields are auto-detected in priority order.
        batch_size: Number of records requested per ChromaDB page.

    Returns:
        Ordered chunk text content as a plain list of strings.

    Raises:
        ChromaPathError: The path is missing or does not look like a ChromaDB dir.
        ChromaCollectionNotFoundError: The named collection does not exist.
        ChromaNamespaceNotFoundError: The namespace has no records.
        ChromaChunkOrderingError: No complete integer ordering field is available.
        ChromaChunkContentError: Retrieved documents are missing or non-string.
    """

    loaded = load_chunks_from_chromadb_with_info(
        chromadb_path=chromadb_path,
        collection_name=collection_name,
        namespace=namespace,
        order_field=order_field,
        batch_size=batch_size,
    )
    logger.info(
        "Loaded %s chunks from ChromaDB collection %r namespace %r ordered by %r",
        loaded.count,
        collection_name,
        namespace,
        loaded.ordering_field,
    )
    return loaded.chunks


def load_chunks_from_chromadb_with_info(
    chromadb_path: str,
    collection_name: str,
    namespace: str,
    order_field: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadedChunks:
    """Return ordered chunks with count and ordering-field metadata."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    db_path = _validate_chromadb_path(chromadb_path)
    client = _persistent_client(str(db_path))
    collection = _get_collection_or_raise(client, collection_name)

    records = _fetch_namespace_records(
        collection=collection,
        namespace=namespace,
        batch_size=batch_size,
    )
    if not records:
        raise ChromaNamespaceNotFoundError(
            f"No records found in collection {collection_name!r} for namespace "
            f"{namespace!r}. Expected metadata field {NAMESPACE_METADATA_FIELD!r} "
            "to match the requested namespace."
        )

    resolved_order_field, ordered_records = _order_records(records, order_field)
    chunks = [record["document"] for record in ordered_records]

    return LoadedChunks(
        chunks=chunks,
        count=len(chunks),
        ordering_field=resolved_order_field,
    )


def _persistent_client(chromadb_path: str) -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise ChromaDependencyError(
                "The chromadb package is required to load chunks. Install it in "
                "this Python environment before calling the chunk loader."
            ) from exc
        raise

    return chromadb.PersistentClient(path=chromadb_path)


def _validate_chromadb_path(chromadb_path: str) -> Path:
    db_path = Path(chromadb_path).expanduser()
    if not db_path.exists():
        raise ChromaPathError(f"ChromaDB path does not exist: {db_path}")
    if not db_path.is_dir():
        raise ChromaPathError(f"ChromaDB path is not a directory: {db_path}")

    marker_names = {path.name for path in db_path.iterdir()}
    has_modern_db = "chroma.sqlite3" in marker_names
    has_legacy_db = {
        "chroma-collections.parquet",
        "chroma-embeddings.parquet",
    }.issubset(marker_names)
    if not has_modern_db and not has_legacy_db:
        raise ChromaPathError(
            f"Path does not appear to be a valid persistent ChromaDB directory: "
            f"{db_path}. Expected chroma.sqlite3, or both legacy files "
            "chroma-collections.parquet and chroma-embeddings.parquet."
        )

    return db_path


def _get_collection_or_raise(client: Any, collection_name: str) -> Any:
    available_names = _list_collection_names(client)
    if collection_name not in available_names:
        available = ", ".join(sorted(available_names)) or "none"
        raise ChromaCollectionNotFoundError(
            f"ChromaDB collection {collection_name!r} was not found. "
            f"Available collections: {available}."
        )

    return client.get_collection(name=collection_name)


def _list_collection_names(client: Any) -> set[str]:
    collections = client.list_collections()
    names: set[str] = set()
    for collection in collections:
        if isinstance(collection, str):
            names.add(collection)
            continue
        name = getattr(collection, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _fetch_namespace_records(
    collection: Any,
    namespace: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    where = {NAMESPACE_METADATA_FIELD: namespace}

    while True:
        result = collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            break
        if len(documents) != len(ids):
            raise ChromaChunkContentError(
                "ChromaDB returned a page with missing document content. "
                f"Expected {len(ids)} documents, got {len(documents)}."
            )
        if len(metadatas) != len(ids):
            raise ChromaChunkContentError(
                "ChromaDB returned a page with missing metadata. "
                f"Expected {len(ids)} metadata entries, got {len(metadatas)}."
            )

        for record_id, document, metadata in zip(ids, documents, metadatas):
            if not isinstance(document, str):
                raise ChromaChunkContentError(
                    f"Record {record_id!r} does not contain string chunk text."
                )
            records.append(
                {
                    "id": record_id,
                    "document": document,
                    "metadata": metadata or {},
                }
            )

        offset += len(ids)
        if len(ids) < batch_size:
            break

    return records


def _order_records(
    records: list[dict[str, Any]],
    order_field: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    if order_field:
        return order_field, _sort_by_field_or_raise(records, order_field)

    for candidate in DEFAULT_ORDER_FIELDS:
        parsed = _parse_order_values(records, candidate)
        if parsed is not None:
            return candidate, _sort_records_with_values(records, parsed)

    keys_found = _metadata_keys_found(records)
    raise ChromaChunkOrderingError(
        "Chunk ordering could not be determined. None of the default metadata "
        f"fields {DEFAULT_ORDER_FIELDS!r} were present with integer or "
        "integer-parseable values on every retrieved record. Metadata keys found "
        f"across records: {keys_found or 'none'}. Provide order_field=... to "
        "explicitly select a metadata field, or fix ingestion metadata."
    )


def _sort_by_field_or_raise(
    records: list[dict[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    parsed = _parse_order_values(records, field_name)
    if parsed is None:
        keys_found = _metadata_keys_found(records)
        raise ChromaChunkOrderingError(
            f"Explicit order field {field_name!r} is missing, inconsistent, or "
            "not integer-parseable on every retrieved record. Metadata keys found "
            f"across records: {keys_found or 'none'}."
        )
    return _sort_records_with_values(records, parsed)


def _parse_order_values(
    records: list[dict[str, Any]],
    field_name: str,
) -> list[int] | None:
    values: list[int] = []
    for record in records:
        metadata = record["metadata"]
        if field_name not in metadata:
            return None

        parsed_value = _parse_int(metadata[field_name])
        if parsed_value is None:
            return None
        values.append(parsed_value)

    return values


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    return None


def _sort_records_with_values(
    records: list[dict[str, Any]],
    order_values: list[int],
) -> list[dict[str, Any]]:
    return [
        record
        for _, record in sorted(
            zip(order_values, records),
            key=lambda item: (item[0], str(item[1]["id"])),
        )
    ]


def _metadata_keys_found(records: list[dict[str, Any]]) -> str:
    keys = sorted(
        {
            key
            for record in records
            for key in record["metadata"].keys()
        }
    )
    return ", ".join(keys)


def _preview(text: str, limit: int = 100) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load ordered text chunks from a local persistent ChromaDB."
    )
    parser.add_argument("chromadb_path", help="Path to the persistent ChromaDB directory")
    parser.add_argument("collection_name", help="ChromaDB collection name")
    parser.add_argument("namespace", help="Namespace metadata value to load")
    parser.add_argument(
        "--order-field",
        help="Explicit metadata field to sort by, bypassing auto-detection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Records to fetch per page (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    loaded = load_chunks_from_chromadb_with_info(
        chromadb_path=args.chromadb_path,
        collection_name=args.collection_name,
        namespace=args.namespace,
        order_field=args.order_field,
        batch_size=args.batch_size,
    )

    print(f"Loaded chunks: {loaded.count}")
    print(f"Ordering field: {loaded.ordering_field}")
    print(f"First chunk preview: {_preview(loaded.chunks[0])}")
    print(f"Last chunk preview: {_preview(loaded.chunks[-1])}")


if __name__ == "__main__":
    main()
