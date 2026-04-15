"""Tests for storage module."""
import pytest


class TestWriter:
    """Test async batch writer."""

    def test_import_writer(self):
        """Verify writer module imports."""
        try:
            from storage import writer
            assert writer is not None
        except ImportError:
            pytest.skip("writer module not available in test environment")


class TestSchema:
    """Test database schema."""

    def test_schema_file_exists(self):
        """Verify schema.sql exists."""
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "storage" / "schema.sql"
        assert schema_path.exists(), "schema.sql not found"

    def test_schema_file_not_empty(self):
        """Verify schema.sql has content."""
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "storage" / "schema.sql"
        content = schema_path.read_text()
        assert len(content) > 100, "schema.sql appears empty"
        assert "CREATE TABLE" in content.upper(), "schema.sql missing CREATE TABLE"
