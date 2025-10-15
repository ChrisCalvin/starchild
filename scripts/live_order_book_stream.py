
import asyncio
import os
from hummingbot_api_client.client import Client
from hummingbot_api_client.async_client import MarketDataRouter

# --- Configuration ---
# It is recommended to use environment variables for credentials
HUMMINGBOT_API_HOST = os.getenv("HUMMINGBOT_API_HOST", "http://localhost:8000")
HUMMINGBOT_API_USERNAME = os.getenv("HUMMINGBOT_API_USERNAME", "admin")
HUMMINGBOT_API_PASSWORD = os.getenv("HUMMINGBOT_API_PASSWORD", "admin")

EXCHANGE = "binance_paper_trade"
TRADING_PAIR = "ETH-USDC"

async def main():
    """
    Connects to the Hummingbot API and streams live order book data.
    """
    print(f"Connecting to Hummingbot API at {HUMMINGBOT_API_HOST}...")

    # 1. Create an authenticated client
    client = Client(
        host=HUMMINGBOT_API_HOST,
        username=HUMMINGBOT_API_USERNAME,
        password=HUMMINGBOT_API_PASSWORD
    )

    # 2. Create a market data router
    market_data_router = MarketDataRouter(client)

    print(f"Starting order book stream for {TRADING_PAIR} on {EXCHANGE}...")
    print("Press Ctrl+C to stop.")

    # 3. Start the order book stream
    try:
        async with market_data_router.get_order_book_stream(exchange=EXCHANGE, trading_pair=TRADING_PAIR) as stream:
            async for order_book_snapshot in stream:
                # The received object is a dictionary representing the order book snapshot
                timestamp = order_book_snapshot.get('timestamp')
                bids = order_book_snapshot.get('bids', [])
                asks = order_book_snapshot.get('asks', [])

                print(f"--- Snapshot at {timestamp} ---")
                if asks:
                    print(f"Best Ask: Price={asks[0][0]}, Amount={asks[0][1]}")
                if bids:
                    print(f"Best Bid: Price={bids[0][0]}, Amount={bids[0][1]}")
                print("--------------------------------====\n")

    except asyncio.CancelledError:
        print("\nStream cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing connection.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
