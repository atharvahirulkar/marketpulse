"""Pytest configuration and shared fixtures."""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_trade_data():
    """Sample trade event from Polygon.io."""
    return {
        "ev": "T",
        "sym": "AAPL",
        "p": 150.25,
        "s": 100,
        "t": 1699000000000,
        "c": [0],
        "i": "12345",
        "x": 1,
    }


@pytest.fixture
def sample_ohlcv_row():
    """Sample OHLCV row."""
    return {
        "symbol": "AAPL",
        "timestamp": 1699000000000,
        "open": 150.0,
        "high": 151.0,
        "low": 149.5,
        "close": 150.5,
        "volume": 1000,
        "vwap": 150.25,
        "trade_count": 50,
    }


@pytest.fixture
def sample_features():
    """Sample engineered features."""
    return {
        "symbol": "AAPL",
        "timestamp": 1699000000000,
        "vwap_5m": 150.25,
        "vwap_15m": 150.10,
        "realized_vol_5m": 0.02,
        "realized_vol_15m": 0.018,
        "price_momentum_1m": 0.003,
        "price_momentum_5m": 0.015,
        "volume_5m": 5000,
        "volume_ratio": 1.2,
        "trade_count_1m": 50,
        "avg_trade_size_1m": 100,
    }
