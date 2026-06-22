# Test Suite

Run the suite from the repository root with:

```bash
PYTHONPATH=src python -m pytest tests
```

These tests use fake Groq/OpenAI-compatible clients and fake embedding models. They
do not require API keys, network access, a real ChromaDB instance, or any
sentence-transformers model download.
