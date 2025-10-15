import abc
import csv
import httpx
import time
import torch
import asyncio
from typing import Iterator, Dict, Any, Optional
from datetime import datetime, timezone

from sasaie_trader.hummingbot_manager import HummingbotManager
from sasaie_trader.config import HummingbotConfig

class DataConnector(abc.ABC):
    """Abstract base class for all data connectors."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the data source."""
        raise NotImplementedError

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        raise NotImplementedError

    @abc.abstractmethod
    async def stream_data(self) -> Iterator[Dict[str, Any]]:
        """Yields data points from the source as a stream."""
        raise NotImplementedError


class CSVDataConnector(DataConnector):
    """A data connector that reads from a local CSV file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._is_connected = False
        self._data_iterator = None

    async def connect(self) -> None:
        """Loads the CSV file and prepares the data iterator."""
        print(f"Connecting to {self.file_path}...")
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as csvfile:
                # Use DictReader to read the CSV into a list of dictionaries
                reader = csv.DictReader(csvfile)
                # Helper to convert numeric strings to numbers
                def _convert_types(row):
                    for key, value in row.items():
                        try:
                            # Attempt to convert to float, then to int if it's a whole number
                            float_val = float(value)
                            if float_val.is_integer():
                                row[key] = int(float_val)
                            else:
                                row[key] = float_val
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            pass
                    return row
                
                self._data_iterator = iter([_convert_types(row) for row in reader])
            self._is_connected = True
            print("Connection successful.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            raise

    async def disconnect(self) -> None:
        """Resets the connector's state."""
        print("Disconnecting...")
        self._is_connected = False
        self._data_iterator = None
        print("Disconnected.")

    async def stream_data(self) -> Iterator[Dict[str, Any]]:
        """Yields rows from the CSV file one by one."""
        if not self._is_connected or self._data_iterator is None:
            raise ConnectionError("Connector is not connected. Call connect() first.")
        
        for data in self._data_iterator:
            yield data


class HummingbotAPIConnector(DataConnector):
    """
    A data connector that streams market data from the Hummingbot API by polling.
    """

    def __init__(self, manager: HummingbotManager, connector_name: str, trading_pair: str, depth: int = 20):
        self.manager = manager
        self.connector_name = connector_name
        self.trading_pair = trading_pair
        self.depth = depth
        self._is_connected = False

    async def connect(self) -> None:
        """Establishes a connection to the Hummingbot API."""
        print(f"Connecting to Hummingbot API for {self.trading_pair}...")
        try:
            await self.manager.init()
            self._is_connected = True
            print("Connection to Hummingbot API successful.")
        except Exception as exc:
            print(f"Error connecting to Hummingbot API: {exc}")
            raise ConnectionError(f"Could not connect to Hummingbot API") from exc

    async def disconnect(self) -> None:
        """Closes the connection to the Hummingbot API."""
        print("Disconnecting from Hummingbot API...")
        if self._is_connected:
            await self.manager.close()
        self._is_connected = False
        print("Disconnected from Hummingbot API.")

    async def stream_data(self) -> Iterator[Dict[str, Any]]:
        """Yields order book data points from the Hummingbot API by polling."""
        if not self._is_connected:
            raise ConnectionError("HummingbotAPIConnector is not connected. Call connect() first.")

        while True:
            try:
                order_book = await self.manager.get_order_book(connector_name=self.connector_name, trading_pair=self.trading_pair, depth=self.depth)
                if order_book:
                    bids = order_book.get("bids", [])[:self.depth]
                    asks = order_book.get("asks", [])[:self.depth]

                    features_list = []
                    for bid in bids:
                        features_list.extend([float(bid["price"]), float(bid["amount"])])
                    for ask in asks:
                        features_list.extend([float(ask["price"]), float(ask["amount"])])
                    
                    expected_feature_len = self.depth * 2 * 2 # depth levels, 2 (bid/ask), 2 (price/amount)
                    if len(features_list) < expected_feature_len:
                        features_list.extend([0.0] * (expected_feature_len - len(features_list)))

                    feature_tensor = torch.tensor(features_list, dtype=torch.float32)

                    yield {
                        "features": feature_tensor,
                        "timestamp": datetime.now(timezone.utc),
                        "metadata": {"trading_pair": self.trading_pair, "data_type": "order_book_snapshot"}
                    }
                await asyncio.sleep(1) # Poll every second
            except httpx.ConnectError as exc:
                # Log the error for debugging, but don't print to console in production
                # logger.error(f"Connection error while streaming data: {exc}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as exc:
                # Log the error for debugging, but don't print to console in production
                # logger.error(f"Error streaming data from Hummingbot API: {exc}")
                await asyncio.sleep(5) # Wait longer on other errors
