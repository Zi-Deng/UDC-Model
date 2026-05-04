"""Dataset profile defaults for NICME experiments."""

from __future__ import annotations

import hashlib
import json
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
    target_recall_classes: tuple[str, ...] = ()
    secondary_target_recall_classes: tuple[str, ...] = ()
    metric_profile: str = "binary"
    split_policy: str = "70_10_10_10"
    cost_matrix_source: str = "dataset"
    critical_pairs: tuple[tuple[str, str, float], ...] = ()
    source_url: str = ""
    cost_reference: str = ""

    @property
    def label2id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.class_names)}

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def cared_classes(self) -> tuple[str, ...]:
        return self.target_recall_classes or (self.target_recall_class,)

    @property
    def cared_class_ids(self) -> tuple[int, ...]:
        label2id = self.label2id
        return tuple(label2id[name] for name in self.cared_classes)

    @property
    def secondary_cared_class_ids(self) -> tuple[int, ...]:
        label2id = self.label2id
        return tuple(label2id[name] for name in self.secondary_target_recall_classes)

    @property
    def critical_pair_ids(self) -> tuple[tuple[int, int, float], ...]:
        label2id = self.label2id
        return tuple((label2id[true_name], label2id[pred_name], cost) for true_name, pred_name, cost in self.critical_pairs)

    @property
    def cost_matrix_sha256(self) -> str:
        payload = json.dumps(
            [[float(value) for value in row] for row in self.cost_matrix],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


EYE_PACS_DR_COST_MATRIX = (
    (0.0, 1.0, 4.0, 9.0, 16.0),
    (1.0, 0.0, 1.0, 4.0, 9.0),
    (4.0, 1.0, 0.0, 1.0, 4.0),
    (9.0, 4.0, 1.0, 0.0, 1.0),
    (16.0, 9.0, 4.0, 1.0, 0.0),
)

EYE_PACS_DR_PROFILE = DatasetProfile(
    name="eyepacs_dr",
    class_names=("DR0", "DR1", "DR2", "DR3", "DR4"),
    target_recall_class="DR4",
    target_recall_classes=("DR4",),
    secondary_target_recall_classes=("DR3", "DR4"),
    cost_matrix=EYE_PACS_DR_COST_MATRIX,
    balanced_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.55,
        accuracy_metric="balanced_accuracy",
    ),
    natural_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.50,
        accuracy_metric="balanced_accuracy",
    ),
    metric_profile="diabetic_retinopathy",
    split_policy="patient_aware_70_10_10_10",
    source_url="https://www.kaggle.com/c/diabetic-retinopathy-detection/data",
    cost_reference="Galdran et al. MICCAI 2020 squared ordinal distance",
)


PMI_CLASS_NAMES = (
    "00378-0208",
    "00378-3855",
    "00591-0461",
    "16729-0020",
    "16729-0168",
    "50111-0434",
    "53489-0156",
    "53746-0544",
    "57664-0377",
    "62037-0831",
    "62037-0832",
    "64380-0803",
    "65162-0253",
    "67253-0901",
    "68382-0008",
    "68382-0227",
    "69097-0127",
    "69097-0128",
    "69315-0904",
    "69315-0905",
)

PMI_CRITICAL_PAIRS = (
    ("50111-0434", "00591-0461", 10.0),
    ("53489-0156", "68382-0227", 10.0),
    ("53746-0544", "00378-0208", 10.0),
    ("68382-0227", "53489-0156", 8.0),
)

PMI_10_NO_CAL_CLASS_NAMES = (
    "00378-0208",
    "00378-3855",
    "00591-0461",
    "16729-0020",
    "50111-0434",
    "53489-0156",
    "53746-0544",
    "62037-0831",
    "68382-0008",
    "68382-0227",
)


def _pmi_cost_matrix(
    class_names: tuple[str, ...],
    critical_pairs: tuple[tuple[str, str, float], ...] = PMI_CRITICAL_PAIRS,
) -> tuple[tuple[float, ...], ...]:
    label2id = {name: idx for idx, name in enumerate(class_names)}
    matrix = [[1.0 for _ in class_names] for _ in class_names]
    for idx in range(len(class_names)):
        matrix[idx][idx] = 0.0
    for true_name, pred_name, cost in critical_pairs:
        if true_name not in label2id or pred_name not in label2id:
            continue
        matrix[label2id[true_name]][label2id[pred_name]] = float(cost)
    return tuple(tuple(row) for row in matrix)


PMI_PILLS_PROFILE = DatasetProfile(
    name="pmi_pills",
    class_names=PMI_CLASS_NAMES,
    target_recall_class="50111-0434",
    target_recall_classes=("50111-0434", "53489-0156", "53746-0544", "68382-0227"),
    cost_matrix=_pmi_cost_matrix(PMI_CLASS_NAMES),
    balanced_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.80,
        accuracy_metric="balanced_accuracy",
    ),
    natural_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.75,
        accuracy_metric="balanced_accuracy",
    ),
    metric_profile="pmi_pills",
    split_policy="official_train_valid_test_valid_split_50_50",
    critical_pairs=PMI_CRITICAL_PAIRS,
    source_url="https://deepblue.lib.umich.edu/data/concern/data_sets/6d56zw997",
    cost_reference="Chen et al. INFORMS Journal on Data Science 2025 Table 5",
)

PMI_10_NO_CAL_CRITICAL_PAIRS = tuple(
    pair
    for pair in PMI_CRITICAL_PAIRS
    if pair[0] in PMI_10_NO_CAL_CLASS_NAMES and pair[1] in PMI_10_NO_CAL_CLASS_NAMES
)

PMI_PILLS_10_NO_CAL_PROFILE = DatasetProfile(
    name="pmi_pills_10_no_cal",
    class_names=PMI_10_NO_CAL_CLASS_NAMES,
    target_recall_class="50111-0434",
    target_recall_classes=("50111-0434", "53489-0156", "53746-0544", "68382-0227"),
    cost_matrix=_pmi_cost_matrix(PMI_10_NO_CAL_CLASS_NAMES, PMI_10_NO_CAL_CRITICAL_PAIRS),
    balanced_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.80,
        accuracy_metric="balanced_accuracy",
    ),
    natural_selection=SelectionDefaults(
        target_recall_floor=0.90,
        accuracy_floor=0.75,
        accuracy_metric="balanced_accuracy",
    ),
    metric_profile="pmi_pills",
    split_policy="official_train_valid_test_no_calibration_60_20_20",
    critical_pairs=PMI_10_NO_CAL_CRITICAL_PAIRS,
    source_url="https://deepblue.lib.umich.edu/data/concern/data_sets/6d56zw997",
    cost_reference="Chen et al. INFORMS Journal on Data Science 2025 Table 5",
)

DATASET_PROFILES = {
    SPIDER_PROFILE.name: SPIDER_PROFILE,
    BREAKHIS_PROFILE.name: BREAKHIS_PROFILE,
    EYE_PACS_DR_PROFILE.name: EYE_PACS_DR_PROFILE,
    PMI_PILLS_PROFILE.name: PMI_PILLS_PROFILE,
    PMI_PILLS_10_NO_CAL_PROFILE.name: PMI_PILLS_10_NO_CAL_PROFILE,
}


def get_profile(name: str) -> DatasetProfile:
    key = name.strip().lower()
    if key not in DATASET_PROFILES:
        raise KeyError(f"Unknown dataset profile {name!r}. Available: {sorted(DATASET_PROFILES)}")
    return DATASET_PROFILES[key]
