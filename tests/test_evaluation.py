from clef_retrieval.evaluation import validate_prediction_shape


def test_validate_prediction_shape_accepts_exactly_five_predictions_per_row():
    assert validate_prediction_shape([["1", "2", "3", "4", "5"], ["a", "b", "c", "d", "e"]])


def test_validate_prediction_shape_rejects_rows_with_wrong_length():
    assert not validate_prediction_shape([["1", "2", "3", "4"], ["a", "b", "c", "d", "e"]])


def test_validate_prediction_shape_rejects_empty_input():
    assert not validate_prediction_shape([])
