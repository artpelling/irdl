"""Pytest fixtures for irdl tests."""

from contextlib import contextmanager

import numpy as np
import pytest
import sofar as sf

from irdl.logging import logger


@pytest.fixture
def progress_tasks(monkeypatch):
    """Record measured work without rendering progress bars."""
    tasks = []

    @contextmanager
    def record(message, total):
        task = {"message": message, "total": total, "completed": 0}
        tasks.append(task)

        def advance(amount):
            task["completed"] += amount
            assert 0 <= task["completed"] <= total

        yield advance

    monkeypatch.setattr(logger, "progress", record)
    return tasks


@pytest.fixture
def sofa_object():
    """Create a minimal SOFA object for testing conversions.

    SOFA convention: positions are (n_positions, 3) where each row is (x, y, z).
    GeneralFIR convention requires Data_Delay field.
    """
    # Create SOFA object with GeneralFIR convention
    sofa = sf.Sofa("GeneralFIR")

    # Set up test data
    sampling_rate = 44100
    n_samples = 256
    n_sources = 2
    n_receivers = 2

    # Create random IR data: shape (n_sources, n_receivers, n_samples)
    rng = np.random.default_rng(42)
    sofa.Data_IR = rng.normal(size=(n_sources, n_receivers, n_samples)).astype(np.float32)

    # Set sampling rate
    sofa.Data_SamplingRate = sampling_rate

    # Set coordinates: shape (n_positions, 3) where each row is (x, y, z)
    # SOFA uses (M, C) or (I, C) where C=3 for (x, y, z)
    sofa.SourcePosition = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)  # Shape: (n_sources, 3)
    sofa.ReceiverPosition = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)  # Shape: (n_receivers, 3)

    # Data_Delay is required for GeneralFIR convention
    sofa.Data_Delay = np.zeros(n_sources, dtype=np.float32)

    return sofa


@pytest.fixture
def sofa_with_optional_fields(sofa_object):
    """Create a SOFA object with optional temperature and humidity fields.

    Note: Temperature and humidity are not part of GeneralFIR convention,
    so we use a different convention that supports them, or skip this test.
    For now, we skip adding these fields as they're not in the base convention.
    """
    # GeneralFIR doesn't support Data_Temperature/Data_Humidity
    # For testing purposes, we just return the base sofa_object
    # The HDF5 test for optional fields will be skipped or use a different approach
    return sofa_object
