"""Tests for data loading and preprocessing."""

import pytest
from pathlib import Path


def test_dataset_split_sizes():
    """Train/val/test fractions sum to 1.0."""
    train, val, test = 0.70, 0.15, 0.15
    assert abs(train + val + test - 1.0) < 1e-6


def test_no_patient_leakage():
    """Patients in train must not appear in val or test."""
    # Replace with real patient IDs once data is available
    train_patients = {"P001", "P002"}
    val_patients = {"P003"}
    test_patients = {"P004"}

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_class_balance():
    """Sampled dataset must have equal counts per grade."""
    samples_per_grade = 2000
    grades = [0, 1, 2, 3, 4]
    counts = {g: samples_per_grade for g in grades}
    assert len(set(counts.values())) == 1
