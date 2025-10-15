"""
Unit tests for the data connectors.
"""

import pytest
import asyncio
from sasaie_trader.connectors import CSVDataConnector

@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_content = "timestamp,value\n2025-01-01T00:00:00Z,100\n2025-01-01T00:01:00Z,101"
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)

def test_csv_connector_happy_path(sample_csv_file):
    """Tests the normal operation of the CSVDataConnector."""
    async def _test():
        # Arrange
        connector = CSVDataConnector(file_path=sample_csv_file)

        # Act
        await connector.connect()
        data_stream = connector.stream_data()
        results = [d async for d in data_stream]
        await connector.disconnect()

        # Assert
        assert len(results) == 2
        assert results[0]["timestamp"] == "2025-01-01T00:00:00Z"
        assert results[0]["value"] == 100
        assert results[1]["timestamp"] == "2025-01-01T00:01:00Z"
        assert results[1]["value"] == 101
    asyncio.run(_test())

def test_csv_connector_stream_before_connect():
    """Tests that streaming before connecting raises an error."""
    async def _test():
        # Arrange
        connector = CSVDataConnector(file_path="dummy.csv")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Connector is not connected"):
            _ = [d async for d in connector.stream_data()]
    asyncio.run(_test())

def test_csv_connector_file_not_found():
    """Tests that a FileNotFoundError is raised for a non-existent file."""
    async def _test():
        # Arrange
        connector = CSVDataConnector(file_path="non_existent_file.csv")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            await connector.connect()
    asyncio.run(_test())
