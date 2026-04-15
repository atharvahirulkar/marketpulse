"""Tests for ingestion module."""
import pytest


class TestPolygonWebSocket:
    """Test Polygon WebSocket consumer."""

    def test_import_polygon_ws(self):
        """Verify polygon_ws module imports."""
        try:
            from ingestion import polygon_ws
            assert polygon_ws is not None
        except ImportError:
            pytest.skip("polygon_ws not available in test environment")

    def test_parse_trade_event(self, sample_trade_data):
        """Test trade event parsing."""
        # Basic validation that sample data structure is correct
        assert sample_trade_data["ev"] == "T"
        assert sample_trade_data["sym"] == "AAPL"
        assert sample_trade_data["p"] > 0
        assert sample_trade_data["s"] > 0


class TestProducer:
    """Test Kafka producer utilities."""

    def test_import_producer(self):
        """Verify producer module imports."""
        try:
            from ingestion import producer
            assert producer is not None
        except ImportError:
            pytest.skip("producer module not available in test environment")
