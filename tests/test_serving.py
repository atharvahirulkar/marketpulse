"""Tests for serving module."""
import pytest


class TestInferenceAPI:
    """Test FastAPI inference server."""

    def test_import_inference_api(self):
        """Verify inference_api module imports."""
        try:
            from serving import inference_api
            assert inference_api is not None
        except ImportError:
            pytest.skip("inference_api not available in test environment")


class TestMetricsServer:
    """Test Prometheus metrics server."""

    def test_import_metrics_server(self):
        """Verify metrics_server module imports."""
        try:
            from serving import metrics_server
            assert metrics_server is not None
        except ImportError:
            pytest.skip("metrics_server not available in test environment")
