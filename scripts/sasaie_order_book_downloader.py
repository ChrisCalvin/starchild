
import os
import csv
from datetime import datetime
from typing import Dict

from hummingbot.core.data_type.common import OrderType
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

class SasaieOrderBookDownloader(ScriptStrategyBase):
    """
    This script downloads order book data for a specific trading pair and saves it to a CSV file.
    It captures a snapshot of the order book at each tick and writes it to a file, creating a new one each day.
    This script is intended to be run from an external directory using `start --script <path_to_this_file>.
    """
    # --- Script Configuration ---
    # Exchange to connect to. You can use a paper trade exchange like 'binance_paper_trade'
    exchange = os.getenv("SAS_EXCHANGE", "binance_paper_trade")
    # Trading pair to download data for
    trading_pair = os.getenv("SAS_TRADING_PAIR", "ETH-USDC")
    # Depth of the order book to capture
    order_book_depth = int(os.getenv("SAS_ORDER_BOOK_DEPTH", 20))
    # How often to write the collected data to the file (in seconds)
    write_interval_seconds = int(os.getenv("SAS_WRITE_INTERVAL", 60))
    # The absolute path to the directory where data will be saved
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_data_dir = os.path.join(_project_root, "data", "hummingbot_data")
    data_directory = os.getenv("SAS_DATA_DIRECTORY", _default_data_dir)

    # --- Internal State ---
    markets = {exchange: {trading_pair}}
    last_write_timestamp = 0
    data_buffer = []
    csv_writer = None
    csv_file = None
    current_date_str = ""

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        os.makedirs(self.data_directory, exist_ok=True)
        self._ensure_file_is_open()

    def on_tick(self):
        """
        Called on each tick of the Hummingbot clock.
        """
        self._ensure_file_is_open()
        self._capture_order_book_snapshot()

        # Check if it's time to write the buffered data to the file
        if self.current_timestamp - self.last_write_timestamp >= self.write_interval_seconds:
            self._flush_data_buffer()

    def _capture_order_book_snapshot(self):
        """
        Fetches the current order book snapshot and adds it to the data buffer.
        """
        order_book = self.connectors[self.exchange].get_order_book(self.trading_pair)
        timestamp = self.current_timestamp

        # Using snapshot for efficiency
        bids = order_book.snapshot[0].head(self.order_book_depth)
        for index, row in bids.iterrows():
            self.data_buffer.append([timestamp, "bid", row["price"], row["amount"]])

        asks = order_book.snapshot[1].head(self.order_book_depth)
        for index, row in asks.iterrows():
            self.data_buffer.append([timestamp, "ask", row["price"], row["amount"]])

    def _flush_data_buffer(self):
        """
        Writes the contents of the data buffer to the CSV file and clears the buffer.
        """
        if self.csv_writer and self.data_buffer:
            buffer_len = len(self.data_buffer)
            self.csv_writer.writerows(self.data_buffer)
            self.data_buffer = []
            self.last_write_timestamp = self.current_timestamp
            self.logger().info(f"Flushed {buffer_len} rows to {self._get_file_path()}")

    def _ensure_file_is_open(self):
        """
        Checks if the CSV file is open and corresponds to the current date.
        If not, it closes the old file and opens a new one.
        """
        current_date = datetime.fromtimestamp(self.current_timestamp).strftime("%Y-%m-%d")
        if current_date != self.current_date_str:
            self._close_file()
            self.current_date_str = current_date
            file_path = self._get_file_path()
            self.logger().info(f"Creating new data file: {file_path}")
            self.csv_file = open(file_path, "a", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            # Write header if the file is new
            if os.path.getsize(file_path) == 0:
                self.csv_writer.writerow(["timestamp", "side", "price", "amount"])

    def _get_file_path(self) -> str:
        """
        Generates the full path for the CSV data file.
        """
        filename = f"order_book_{self.exchange}_{self.trading_pair.replace('-', '')}_{self.current_date_str}.csv"
        return os.path.join(self.data_directory, filename)

    def _close_file(self):
        """
        Closes the currently open CSV file if it exists.
        """
        if self.csv_file:
            self.csv_file.close()
            self.csv_writer = None
            self.csv_file = None

    def on_stop(self):
        """
        Called when the script is stopped. Ensures any remaining data is flushed.
        """
        self._flush_data_buffer()
        self._close_file()
        self.logger().info("Order book downloader script stopped.")
