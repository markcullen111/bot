# manipulation_detector.py


import numpy as np

class MarketManipulationDetector:
    """Detects stop hunts, flash crashes, spoofing, & liquidity traps"""

    def __init__(self):
        pass

    def detect_stop_hunt(self, df):
        """
        Detects stop hunts by checking sudden wicks below support or above resistance.
        """
        wick_threshold = 0.015  # 1.5% wick is suspicious
        last_candle = df.iloc[-1]

        wick_down = (last_candle["low"] < last_candle["open"] * (1 - wick_threshold))
        wick_up = (last_candle["high"] > last_candle["open"] * (1 + wick_threshold))

        return wick_down, wick_up

    def detect_flash_crash(self, df):
        """
        Detects flash crashes (huge red candles with low volume)
        """
        last_candle = df.iloc[-1]
        body_size = abs(last_candle["close"] - last_candle["open"])
        wick_size = last_candle["high"] - last_candle["low"]
        volume = last_candle["volume"]
        
        crash_detected = (body_size > wick_size * 0.8) and (volume < df["volume"].rolling(20).mean().iloc[-1] * 0.5)
        return crash_detected

    def detect_spoofing(self, order_book):
        """
        Detects order spoofing (large orders that suddenly disappear)
        """
        bids = np.array(order_book["bids"])
        asks = np.array(order_book["asks"])

        bid_spoofing = np.any(bids[:, 1] > np.mean(bids[:, 1]) * 4)
        ask_spoofing = np.any(asks[:, 1] > np.mean(asks[:, 1]) * 4)

        return bid_spoofing, ask_spoofing

    def detect_fat_finger_trade(self, df):
        """
        Detects "fat-finger" trades (abnormal large single trade movement)
        """
        avg_candle_size = abs(df["close"] - df["open"]).rolling(20).mean()
        last_candle = df.iloc[-1]
        
        return abs(last_candle["close"] - last_candle["open"]) > avg_candle_size.iloc[-1] * 3

