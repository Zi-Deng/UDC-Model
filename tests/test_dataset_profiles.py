from nicme.dataset_profiles import EYE_PACS_DR_PROFILE, PMI_PILLS_10_NO_CAL_PROFILE, PMI_PILLS_PROFILE, get_profile


def test_eyepacs_dr_profile_uses_galdran_quadratic_cost_matrix():
    assert EYE_PACS_DR_PROFILE.num_classes == 5
    assert EYE_PACS_DR_PROFILE.class_names == ("DR0", "DR1", "DR2", "DR3", "DR4")
    assert EYE_PACS_DR_PROFILE.cost_matrix[0] == (0.0, 1.0, 4.0, 9.0, 16.0)
    assert EYE_PACS_DR_PROFILE.cost_matrix[4] == (16.0, 9.0, 4.0, 1.0, 0.0)
    assert EYE_PACS_DR_PROFILE.cared_class_ids == (4,)
    assert EYE_PACS_DR_PROFILE.secondary_cared_class_ids == (3, 4)
    assert len(EYE_PACS_DR_PROFILE.cost_matrix_sha256) == 64


def test_pmi_profile_has_asymmetric_critical_pair_costs():
    profile = PMI_PILLS_PROFILE
    label2id = profile.label2id
    matrix = profile.cost_matrix

    assert profile.num_classes == 20
    assert matrix[label2id["50111-0434"]][label2id["00591-0461"]] == 10.0
    assert matrix[label2id["53489-0156"]][label2id["68382-0227"]] == 10.0
    assert matrix[label2id["53746-0544"]][label2id["00378-0208"]] == 10.0
    assert matrix[label2id["68382-0227"]][label2id["53489-0156"]] == 8.0
    assert matrix[label2id["00591-0461"]][label2id["50111-0434"]] == 1.0
    assert profile.cared_class_ids == (
        label2id["50111-0434"],
        label2id["53489-0156"],
        label2id["53746-0544"],
        label2id["68382-0227"],
    )


def test_get_profile_includes_multiclass_profiles():
    assert get_profile("eyepacs_dr") is EYE_PACS_DR_PROFILE
    assert get_profile("pmi_pills") is PMI_PILLS_PROFILE
    assert get_profile("pmi_pills_10_no_cal") is PMI_PILLS_10_NO_CAL_PROFILE


def test_pmi_10_no_cal_profile_has_fixed_subset_and_costs():
    profile = PMI_PILLS_10_NO_CAL_PROFILE
    label2id = profile.label2id
    matrix = profile.cost_matrix

    assert profile.class_names == (
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
    assert profile.num_classes == 10
    assert profile.split_policy == "official_train_valid_test_no_calibration_60_20_20"
    assert len(matrix) == 10
    assert all(len(row) == 10 for row in matrix)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if row_idx == col_idx:
                assert value == 0.0
            elif (profile.class_names[row_idx], profile.class_names[col_idx]) not in {
                ("50111-0434", "00591-0461"),
                ("53489-0156", "68382-0227"),
                ("53746-0544", "00378-0208"),
                ("68382-0227", "53489-0156"),
            }:
                assert value == 1.0
    assert matrix[label2id["50111-0434"]][label2id["00591-0461"]] == 10.0
    assert matrix[label2id["53489-0156"]][label2id["68382-0227"]] == 10.0
    assert matrix[label2id["53746-0544"]][label2id["00378-0208"]] == 10.0
    assert matrix[label2id["68382-0227"]][label2id["53489-0156"]] == 8.0
