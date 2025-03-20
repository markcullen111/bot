# whale_tracker.py


import requests
import numpy as np
import time

class WhaleTracker:
    """Tracks large on-chain transactions & exchange inflows/outflows"""

    def __init__(self, symbol="BTC", exchange="binance"):
        self.symbol = symbol
        self.exchange = exchange
        self.large_tx_threshold = 1000  # Track transactions > 1000 BTC

    def fetch_onchain_transactions(self):
        """
        Retrieves large BTC transactions from Whale Alert API (or an on-chain API).
        """
        url = f"https://api.whale-alert.io/v1/transactions?api_key=YOUR_API_KEY&symbol={self.symbol.lower()}"
        try:
            response = requests.get(url)
            data = response.json()
            large_txs = [tx for tx in data["transactions"] if tx["amount"] > self.large_tx_threshold]
            return large_txs
        except Exception as e:
            print(f"Error fetching whale transactions: {e}")
            return []

    def fetch_exchange_flows(self):
        """
        Retrieves exchange inflow/outflow data (Binance, Coinbase, etc.).
        """
        url = f"https://api.cryptoquant.com/v1/eth/address-flows/{self.exchange}"
        try:
            response = requests.get(url)
            data = response.json()
            return data
        except Exception as e:
            print(f"Error fetching exchange flows: {e}")
            return {}

    def detect_whale_activity(self):
        """
        Detects whale accumulation or distribution based on transaction data.
        """
        large_txs = self.fetch_onchain_transactions()
        exchange_flows = self.fetch_exchange_flows()

        if len(large_txs) > 3:
            print("🐳 Large Whale Transactions Detected!")

        # Detect large exchange deposits (potential selling pressure)
        if exchange_flows.get("inflow", 0) > 5000:
            print("⚠️ Large Exchange Inflows Detected! Possible Selling Pressure.")

        # Detect large exchange withdrawals (potential accumulation)
        if exchange_flows.get("outflow", 0) > 5000:
            print("🟢 Large Exchange Outflows Detected! Possible Accumulation.")

        return {
            "whale_transactions": large_txs,
            "exchange_flows": exchange_flows
        }

