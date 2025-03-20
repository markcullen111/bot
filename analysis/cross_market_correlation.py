# cross_market_correlation.py


import numpy as np
import pandas as pd

class CrossMarketCorrelation:
    """
    Tracks the correlation between BTC and altcoins to predict price movements.
    """

    def __init__(self):
        self.correlation_threshold = 0.8  # Minimum correlation for prediction

    def calculate_correlation(self, btc_df, alt_df):
        """
        Calculates rolling correlation between BTC and an altcoin.
        """
        merged_df = btc_df[["close"]].rename(columns={"close": "btc_close"}).join(
            alt_df[["close"]].rename(columns={"close": "alt_close"}), how="inner"
        )

        # Rolling correlation over 50 periods
        merged_df["correlation"] = merged_df["btc_close"].rolling(50).corr(merged_df["alt_close"])
        return merged_df

    def predict_altcoin_movement(self, btc_df, alt_df):
        """
        Uses BTC momentum to predict altcoin moves.
        """
        correlation_df = self.calculate_correlation(btc_df, alt_df)
        last_correlation = correlation_df["correlation"].iloc[-1]
        btc_momentum = btc_df["close"].iloc[-1] - btc_df["close"].iloc[-5]

        if last_correlation > self.correlation_threshold and btc_momentum > 0:
            print("🚀 BTC is bullish, high correlation detected—buy altcoin")
            return "buy"
        elif last_correlation > self.correlation_threshold and btc_momentum < 0:
            print("⚠️ BTC is bearish, high correlation detected—avoid altcoin longs")
            return "sell"
        else:
            print("🔍 BTC and altcoin correlation is weak—no trade signal.")
            return "hold"

