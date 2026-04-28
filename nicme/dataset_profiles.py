"""Dataset profile defaults for binary-first NICME experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionDefaults:
    target_recall_floor: float
    accuracy_floor: float
    accuracy_metric: str = "accuracy"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    class_names: tuple[str, ...]
    target_recall_class: str
    cost_matrix: tuple[tuple[float, ...], ...]
    balanced_selection: SelectionDefaults
    natural_selection: SelectionDefaults

    @property
    def label2id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.class_names)}


SPIDER_PROFILE = DatasetProfile(
    name="spider",
    class_names=("black_widow", "false_widow"),
    target_recall_class="black_widow",
    cost_matrix=((0.0, 10.0), (1.0, 0.0)),
    balanced_selection=SelectionDefaults(target_recall_floor=0.95, accuracy_floor=0.80),
    natural_selection=SelectionDefaults(
        target_recall_floor=0.95,
        accuracy_floor=0.75,
        accuracy_metric="balanced_accuracy",
    ),
)

BREAKHIS_PROFILE = DatasetProfile(
    name="breakhis",
    class_names=("benign", "malignant"),
    target_recall_class="malignant",
    cost_matrix=((0.0, 1.0), (10.0, 0.0)),
    balanced_selection=SelectionDefaults(target_recall_floor=0.97, accuracy_floor=0.85),
    natural_selection=SelectionDefaults(
        target_recall_floor=0.97,
        accuracy_floor=0.80,
        accuracy_metric="balanced_accuracy",
    ),
)

DATASET_PROFILES = {
    SPIDER_PROFILE.name: SPIDER_PROFILE,
    BREAKHIS_PROFILE.name: BREAKHIS_PROFILE,
}


def get_profile(name: str) -> DatasetProfile:
    key = name.strip().lower()
    if key not in DATASET_PROFILES:
        raise KeyError(f"Unknown dataset profile {name!r}. Available: {sorted(DATASET_PROFILES)}")
    return DATASET_PROFILES[key]
