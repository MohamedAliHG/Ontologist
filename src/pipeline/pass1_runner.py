from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from config.settings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_GROQ_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    config_get,
    load_pass1_config,
)
from consolidation.class_consolidator import consolidate_raw_classes
from export.schema_profile_exporter import export_schema_profile
from extraction.chunk_llm_extractor import extract_schema_candidates_for_chunk
from models import PipelineState


logger = logging.getLogger(__name__)


def process_chunks(
    chunks: list[str],
    threshold: float,
    *,
    extraction_kwargs: dict[str, Any] | None = None,
) -> PipelineState:
    """Process chunks in order into a fully populated PipelineState."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    state = PipelineState()
    total_chunks = len(chunks)
    extractor_options = extraction_kwargs or {}

    for chunk_idx, chunk in enumerate(chunks):
        extraction = extract_schema_candidates_for_chunk(
            chunk=chunk,
            chunk_idx=chunk_idx,
            state=state,
            **extractor_options,
        )

        state.raw_entities.extend(extraction.raw_entities)
        state.raw_relationships.extend(extraction.raw_relationships)

        updated_classes, consolidation_entries = consolidate_raw_classes(
            new_raw_classes=extraction.raw_classes,
            consolidated_classes=state.consolidated_classes,
            threshold=threshold,
        )
        state.consolidated_classes = updated_classes
        state.consolidation_log.extend(consolidation_entries)

        state.raw_classes.extend(extraction.raw_classes)

        decision_counts = _count_decisions_by_type(consolidation_entries)
        state.chunk_log.append(
            {
                "chunk_idx": chunk_idx,
                "raw_classes_extracted": len(extraction.raw_classes),
                "raw_entities_extracted": len(extraction.raw_entities),
                "raw_relationships_extracted": len(extraction.raw_relationships),
                "consolidation_decisions": decision_counts,
            }
        )

        logger.info(
            "chunk %s/%s: added raw classes=%s, entities=%s, relationships=%s; "
            "running totals consolidated_classes=%s, raw_entities=%s, "
            "raw_relationships=%s",
            chunk_idx + 1,
            total_chunks,
            len(extraction.raw_classes),
            len(extraction.raw_entities),
            len(extraction.raw_relationships),
            len(state.consolidated_classes),
            len(state.raw_entities),
            len(state.raw_relationships),
        )

    return state


def _count_decisions_by_type(consolidation_entries: list[Any]) -> dict[str, int]:
    counts = {
        "exact_match": 0,
        "embedding_merge": 0,
        "new_class": 0,
    }
    for entry in consolidation_entries:
        decision_type = getattr(entry, "decision_type")
        counts[decision_type] = counts.get(decision_type, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Pass 1 schema-candidate extraction and class consolidation over "
            "ordered chunks loaded from a local persistent ChromaDB."
        )
    )
    parser.add_argument(
        "--config",
        help="Optional TOML config file, for example config/pass1.default.toml",
    )
    parser.add_argument("chromadb_path", help="Path to the persistent ChromaDB directory")
    parser.add_argument("collection_name", help="ChromaDB collection name")
    parser.add_argument("namespace", help="Namespace metadata value to load")
    parser.add_argument("output_dir", help="Directory where final schema/export files are written")
    parser.add_argument(
        "--order-field",
        help="Explicit metadata field passed through to the Chroma chunk loader",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Cosine similarity threshold for class embedding merges "
            f"(default: {DEFAULT_SIMILARITY_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for progress output (default: INFO)",
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "openai", "local"),
        help="LLM client provider passed to the per-chunk extractor (default: groq)",
    )
    parser.add_argument(
        "--model",
        help="LLM model name passed to the per-chunk extractor",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL for local or custom endpoints",
    )
    parser.add_argument(
        "--embedding-model",
        help=(
            "Sentence-transformers model for class consolidation "
            f"(default: {DEFAULT_EMBEDDING_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--document-id",
        help="Document id to write into schema_profile.json; defaults to a UUID",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    run_config = load_pass1_config(args.config)
    order_field = args.order_field or config_get(run_config, "chroma", "order_field")
    threshold = (
        args.threshold
        if args.threshold is not None
        else config_get(
            run_config,
            "consolidation",
            "threshold",
            DEFAULT_SIMILARITY_THRESHOLD,
        )
    )
    provider = args.provider or config_get(run_config, "llm", "provider", "groq")
    model = args.model or config_get(run_config, "llm", "model", DEFAULT_GROQ_MODEL)
    base_url = args.base_url or config_get(run_config, "llm", "base_url")
    embedding_model = args.embedding_model or config_get(
        run_config,
        "consolidation",
        "embedding_model",
        DEFAULT_EMBEDDING_MODEL_NAME,
    )
    os.environ["SCHEMA_CLASS_EMBEDDING_MODEL"] = embedding_model

    from loading.chroma_chunk_loader import load_chunks_from_chromadb

    chunks = load_chunks_from_chromadb(
        chromadb_path=args.chromadb_path,
        collection_name=args.collection_name,
        namespace=args.namespace,
        order_field=order_field,
    )
    extraction_kwargs = {
        key: value
        for key, value in {
            "provider": provider,
            "model": model,
            "base_url": base_url,
        }.items()
        if value is not None
    }
    state = process_chunks(
        chunks=chunks,
        threshold=threshold,
        extraction_kwargs=extraction_kwargs,
    )
    export_result = export_schema_profile(
        state=state,
        output_dir=args.output_dir,
        document_id=args.document_id,
    )

    print(f"Processed chunks: {len(chunks)}")
    print(f"Document id: {export_result.document_id}")
    print(f"Output directory: {export_result.output_dir}")
    print(f"Schema JSON: {export_result.schema_path}")
    for artifact_name, count in sorted(export_result.counts.items()):
        print(f"{artifact_name}: {count}")


if __name__ == "__main__":
    main()
