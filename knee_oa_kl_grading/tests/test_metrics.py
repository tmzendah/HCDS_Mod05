"""Tests for evaluation metrics."""

import pytest


def test_mae_perfect_predictions():
    from sklearn.metrics import mean_absolute_error
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    assert mean_absolute_error(y_true, y_pred) == 0.0


def test_mae_off_by_one():
    from sklearn.metrics import mean_absolute_error
    y_true = [0, 1, 2, 3, 4]
    y_pred = [1, 2, 3, 4, 4]
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.8)


def test_balanced_accuracy_uniform():
    from sklearn.metrics import balanced_accuracy_score
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(1.0)


def test_macro_f1_perfect():
    from sklearn.metrics import f1_score
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    assert f1_score(y_true, y_pred, average="macro") == pytest.approx(1.0)
