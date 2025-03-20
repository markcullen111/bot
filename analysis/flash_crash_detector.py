# flash_crash_detector.py


import numpy as np
import ccxt
import time

class FlashCrashDetector:
    """AI-driven detection of flash crashes, liquidation cascades & stop hunts"""

    def __init__(self, symbol="BTC/USDT"):
        self.symbol = symbol
        self.exchange = ccxt.binance()  # Change to your preferred exchange
        self.recent_liquidations = []

    def fetch_order_book(self):
        """
        Retrieves real-time order book data to detect spoofing & bid-ask imbalances.
        """
        order_book = self.exchange.fetch_order_book(self.symbol, limit=10)
        bids = np.array(order_book["bids"])
        asks = np.array(order_book["asks"])

        # Detect bid-ask imbalance
        bid_pressure = np.sum(bids[:, 1]) / (np.sum(asks[:, 1]) + 1e-6)
        ask_pressure = np.sum(asks[:, 1]) / (np.sum(bids[:, 1]) + 1e-6)

        return bid_pressure, ask_pressure

    def fetch_liquidation_data(self):
        """
        Retrieves real-time liquidation data from Binance Futures API.
        """
        url = f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={self.symbol.replace('/', '')}"
        try:
            liquidations = self.exchange.request(url, method="GET")
            return liquidations
        except Exception as e:
            print(f"Error fetching liquidations: {e}")
            return []

    def detect_flash_crash(self, df):
        """
        Detects flash crashes by analyzing price action & order flow.
        """
        last_row = df.iloc[-1]

        # Calculate volatility (ATR-based)
        df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
        volatility = df["atr"].iloc[-1] / last_row["close"]

        # Detect sudden price drops (Flash Crash)
        df["price_change"] = df["close"].pct_change()
        flash_crash_detected = df["price_change"].iloc[-1] < -0.03  # 3%+ drop in 1 candle

        # Detect liquidation cascades
        liquidations = self.fetch_liquidation_data()
        if len(liquidations) > 5:  # High liquidation activity
            print("⚠️ Liquidation Cascade Detected!")
            return True

        # Detect Stop Hunts (Wick below support)
        wick_down = last_row["low"] < df["low"].rolling(20).min().iloc[-1] * 0.98
        if wick_down:
            print("⚠️ Stop Hunt Detected!")

        return flash_crash_detected or wick_down

