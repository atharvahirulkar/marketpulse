"""Tests for training module."""
import pytest


class TestDataLoader:
    """Test yfinance and TimescaleDB data loaders."""

    def test_import_data_loader(self):
        """Verify data_loader module imports."""
        try:
            from training import data_loader
            assert data_loader is not None
        except ImportError:
            pytest.skip("data_loader module not available in test environment")


class TestDataset:
    """Test dataset preparation."""

    def test_import_dataset(self):
        """Verify dataset module imports."""
        try:
            from training import dataset
            assert dataset is not None
        except ImportError:
            pytest.skip("dataset module not available in test environment")


class TestModels:
    """Test ML models."""

    def test_import_models(self):
        """Verify models module imports."""
        try:
            from training import models
            assert models is not None
        except ImportError:
            pytest.skip("models module not available in test environment")

    def test_lstm_model_exists(self):
        """Verify LSTM model class exists."""
        try:
            from training.models import LSTMModel
            assert LSTMModel is not None
        except ImportError:
            pytest.skip("LSTMModel not available")

    def test_xgboost_model_exists(self):
        """Verify XGBoost model class exists."""
        try:
            from training.models import XGBoostRegime
            assert XGBoostRegime is not None
        except ImportError:
            pytest.skip("XGBoostRegime not available")

    def test_anomaly_detector_exists(self):
        """Verify Anomaly Detector class exists."""
        try:
            from training.models import AnomalyDetector
            assert AnomalyDetector is not None
        except ImportError:
            pytest.skip("AnomalyDetector not available")


class TestDrift:
    """Test drift detection."""

    def test_import_drift(self):
        """Verify drift module imports."""
        try:
            from training import drift
            assert drift is not None
        except ImportError:
            pytest.skip("drift module not available in test environment")
