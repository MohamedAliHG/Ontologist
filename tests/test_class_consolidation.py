from __future__ import annotations

import copy
import math

import pytest

from consolidation import class_consolidator
from models import ConsolidatedClass, RawClass


def raw_class(
    class_id: str = "person_candidate",
    name: str = "Person",
    description: str = "A human actor.",
    chunk_idx: int = 1,
) -> RawClass:
    return RawClass(
        id=class_id,
        name=name,
        description=description,
        chunk_idx=chunk_idx,
    )


def test_exact_match_merge_does_not_call_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(model_name: str) -> object:
        raise AssertionError("Exact-match merge should not embed the raw class")

    monkeypatch.setattr(class_consolidator, "initialize_embedding_model", fail_if_called)
    new_classes = [raw_class(name="person")]
    before = copy.deepcopy(new_classes)
    consolidated = {
        "person": ConsolidatedClass(
            id="person",
            canonical_name="Person",
            description="A human actor.",
            aliases=["Human"],
            mention_count=2,
            embedding=[1.0, 0.0],
            source_chunk_indices=[0],
        )
    }

    updated, decisions = class_consolidator.consolidate_raw_classes(
        new_raw_classes=new_classes,
        consolidated_classes=consolidated,
        threshold=0.82,
    )

    assert new_classes == before
    assert list(updated) == ["person"]
    assert updated["person"].mention_count == 3
    assert updated["person"].aliases == ["Human", "person"]
    assert updated["person"].embedding == [1.0, 0.0]
    assert decisions[0].decision_type == "exact_match"
    assert decisions[0].similarity_score is None


def test_embedding_similarity_above_threshold_merges_with_running_mean(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    # New vector has cosine similarity 0.8 with [1.0, 0.0].
    new_vector = [0.8, 0.6]
    fake_model = fake_embedding_model_factory({"Automobile.": new_vector})
    monkeypatch.setattr(
        class_consolidator,
        "initialize_embedding_model",
        lambda model_name: fake_model,
    )
    consolidated = {
        "vehicle": ConsolidatedClass(
            id="vehicle",
            canonical_name="Vehicle",
            description="A conveyance.",
            aliases=["Vehicle"],
            mention_count=3,
            embedding=[1.0, 0.0],
            source_chunk_indices=[0, 1, 2],
        )
    }

    updated, decisions = class_consolidator.consolidate_raw_classes(
        new_raw_classes=[
            raw_class(
                class_id="automobile",
                name="Automobile",
                description="A machine for transport.",
                chunk_idx=3,
            )
        ],
        consolidated_classes=consolidated,
        threshold=0.75,
    )

    # Running mean used by the implementation:
    # [((old_value * old_mention_count) + new_value) / (old_mention_count + 1)].
    mean = [((1.0 * 3) + 0.8) / 4, ((0.0 * 3) + 0.6) / 4]
    norm = math.sqrt(sum(value * value for value in mean))
    expected = [value / norm for value in mean]
    assert updated["vehicle"].embedding == pytest.approx(expected)
    assert updated["vehicle"].mention_count == 4
    assert decisions[0].decision_type == "embedding_merge"
    assert decisions[0].similarity_score == pytest.approx(0.8)


def test_similarity_below_threshold_creates_new_record(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    # Similarity to [1.0, 0.0] is 0.2, below the 0.9 threshold.
    fake_model = fake_embedding_model_factory({"Policy.": [0.2, math.sqrt(0.96)]})
    monkeypatch.setattr(
        class_consolidator,
        "initialize_embedding_model",
        lambda model_name: fake_model,
    )
    consolidated = {
        "person": ConsolidatedClass(
            id="person",
            canonical_name="Person",
            description="A human actor.",
            aliases=["Person"],
            mention_count=1,
            embedding=[1.0, 0.0],
            source_chunk_indices=[0],
        )
    }

    updated, decisions = class_consolidator.consolidate_raw_classes(
        new_raw_classes=[raw_class(class_id="policy", name="Policy", description="A rule.")],
        consolidated_classes=consolidated,
        threshold=0.9,
    )

    assert sorted(updated) == ["person", "policy"]
    assert updated["policy"].mention_count == 1
    assert decisions[0].decision_type == "new_class"
    assert decisions[0].similarity_score == pytest.approx(0.2)


def test_similarity_at_threshold_boundary_merges_because_comparison_is_greater_equal(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_model_factory: object,
) -> None:
    # Similarity is exactly 0.8 against [1.0, 0.0]; implementation uses >=.
    fake_model = fake_embedding_model_factory({"Automobile.": [0.8, 0.6]})
    monkeypatch.setattr(
        class_consolidator,
        "initialize_embedding_model",
        lambda model_name: fake_model,
    )
    consolidated = {
        "vehicle": ConsolidatedClass(
            id="vehicle",
            canonical_name="Vehicle",
            description="A conveyance.",
            aliases=["Vehicle"],
            mention_count=1,
            embedding=[1.0, 0.0],
            source_chunk_indices=[0],
        )
    }

    updated, decisions = class_consolidator.consolidate_raw_classes(
        new_raw_classes=[
            raw_class(name="Automobile", description="A vehicle.", chunk_idx=1)
        ],
        consolidated_classes=consolidated,
        threshold=0.8,
    )

    assert list(updated) == ["vehicle"]
    assert decisions[0].decision_type == "embedding_merge"
    assert decisions[0].threshold == 0.8
