# Ontologist

ontology/schema candidate generation pipeline for GraphRAG knowledge graph
construction.

This project reads ordered text chunks from an existing local ChromaDB collection,
uses an LLM to extract schema candidates chunk by chunk, progressively consolidates
class and relationship-type candidates, and exports a SchemaProfile-shaped JSON file
for a later constrained extraction pass.

## What This Pipeline Does

- Loads chunks from a local persistent ChromaDB directory.
- Sends each chunk to an OpenAI-compatible LLM client, with current schema context.
- Appends raw classes, entities, and relationships as permanent audit logs.
- Consolidates class candidates with exact matching and local embeddings.
- Consolidates relationship types by domain/range pair with exact matching and local
  embeddings.
- Exports `allowed_nodes`, `allowed_relationships`, and strict
  `[domain, relationship_type, range]` triples.

This is not the final knowledge graph extraction step. It produces schema
candidates for a downstream LangChain `LLMGraphTransformer` run.

## Repository Layout

```text
config/
  pass1.default.toml          Default Pass 1 configuration
scripts/
  run_pass1.py                CLI entry point from the repo root
src/
  config/                     Config loading and defaults
  consolidation/              Class consolidation logic
  export/                     SchemaProfile and audit exports
  extraction/                 Per-chunk LLM extraction
  loading/                    Local ChromaDB chunk loading
  models/                     Dataclasses for pipeline state
  pipeline/                   Pass 1 orchestration loop
  prompts/                    LLM prompt templates
outputs/
  .gitkeep                    Output directory placeholder
```

## Setup

Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a local `.env` or export your API key in the shell:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

For a local OpenAI-compatible endpoint, use `--provider local --base-url ...`
instead of Groq.

## Configuration

Default config lives in:

```bash
config/pass1.default.toml
```

Important settings:

- `llm.model`: LLM model name.
- `llm.temperature`: LLM sampling temperature.
- `consolidation.threshold`: embedding similarity threshold for class and
  relationship-type merges.
- `consolidation.embedding_model`: local sentence-transformers model.
- `chroma.order_field`: metadata field used to deterministically order chunks.
- `chroma.batch_size`: Chroma records fetched per page.

CLI arguments override config values.

## Run Pass 1

```bash
python scripts/run_pass1.py \
  --config config/pass1.default.toml \
  chroma_db \
  collection_demo2 \
  hybrid_test \
  outputs/pass1_groq \
  --order-field chunk_index \
  --threshold 0.82 \
  --model llama-3.3-70b-versatile \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --document-id demo_document
```

Argument order after options:

```text
chromadb_path collection_name namespace output_dir
```

The ChromaDB directory must already exist and contain records for the requested
collection and namespace. Ingestion and chunking are handled outside this pipeline.

## Outputs

The output directory receives:

- `schema_profile.json`
- `raw_classes.json` / `raw_classes.csv`
- `consolidated_classes.json` / `consolidated_classes.csv`
- `raw_entities.json` / `raw_entities.csv`
- `raw_relationships.json` / `raw_relationships.csv`

`schema_profile.json` includes:

- `document_id`
- `generated_at`
- `allowed_nodes`
- `allowed_relationships`
- `strict_relationships`
- `class_provenance`
- `relationship_type_provenance`
- `consolidation_log`


## Notes

- Raw classes, entities, and relationships are append-only audit trails.
- Entities are intentionally not consolidated in Pass 1.
- Class resolution during export follows multi-hop consolidation chains to final
  canonical class ids.
