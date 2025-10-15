import logging
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# Configure logger to log to console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollectorStrategy(ScriptStrategyBase):
    """
    A simple strategy that connects to a market and does nothing.
    Its sole purpose is to be active on a market, which causes Hummingbot
    to generate order book snapshot events, which are then published by the MQTT bridge.
    """

    def __init__(self, connectors: dict, exchange: str, market: str):
        super().__init__(connectors)
        try:
            with open("/home/hummingbot/logs/debug_strategy.log", "w") as f:
                f.write(f"Strategy Initializing...\n")
                f.write(f"Exchange: {exchange}\n")
                f.write(f"Market: {market}\n")
            self.exchange = exchange
            self.trading_pair = market
            self.markets = {self.exchange: {self.trading_pair}}
            logging.info("DataCollectorStrategy: Initialized.")
            logging.info(f"DataCollectorStrategy: Markets: {self.markets}")
        except Exception as e:
            with open("/home/hummingbot/logs/debug_strategy.log", "w") as f:
                f.write(f"Error during __init__: {e}\n")

    def on_tick(self):
        # Do nothing. The strategy is just here to keep the connection alive.
        logging.info("DataCollectorStrategy: on_tick")
        pass

    def format_status(self) -> str:
        """
        Returns a string containing the current status of the strategy.
        """
        status = f"Data Collector Strategy is running for {self.trading_pair} on {self.exchange}."
        logging.info(f"DataCollectorStrategy: format_status: {status}")
        return status