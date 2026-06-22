from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def no_sleep() -> Any:
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)

    sleep.calls = calls
    return sleep


class FakeEmbeddingModel:
    def __init__(self, vectors_by_prefix: dict[str, list[float]]) -> None:
        self.vectors_by_prefix = vectors_by_prefix
        self.encoded_texts: list[str] = []

    def encode(self, text: str, normalize_embeddings: bool = True) -> list[float]:
        self.encoded_texts.append(text)
        for prefix, vector in self.vectors_by_prefix.items():
            if text.startswith(prefix):
                return vector
        raise AssertionError(f"No fake embedding vector configured for {text!r}")


@pytest.fixture
def fake_embedding_model_factory() -> Any:
    def factory(vectors_by_prefix: dict[str, list[float]]) -> FakeEmbeddingModel:
        return FakeEmbeddingModel(vectors_by_prefix)

    return factory


def llm_response(content: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class FakeLLMCompletions:
    def __init__(self, side_effects: list[Any]) -> None:
        self.side_effects = list(side_effects)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.side_effects:
            raise AssertionError("Fake LLM client received more calls than expected")
        next_effect = self.side_effects.pop(0)
        if isinstance(next_effect, BaseException):
            raise next_effect
        return llm_response(next_effect)


class FakeLLMClient:
    def __init__(self, side_effects: list[Any]) -> None:
        self.completions = FakeLLMCompletions(side_effects)
        self.chat = SimpleNamespace(completions=self.completions)
