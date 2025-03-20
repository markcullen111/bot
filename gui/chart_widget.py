# gui/widgets/chart_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                           QLabel, QPushButton, QCheckBox, QGridLayout, QFrame)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np
from datetime import datetime, timedelta

class ChartWidget(QWidget):
    """
    Interactive chart widget for displaying market data with
    technical indicators and custom annotations.
    """
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.current_symbol = None
        self.current_timeframe = None
        self.current_data = None
        self.indicators = []
        
        # Initialize UI
        self._init_ui()
        
        # Update timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Symbol selector
        controls_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        if hasattr(self.trading_system, 'config'):
            symbols = self.trading_system.config.get("trading", {}).get("symbols", [])
            self.symbol_combo.addItems(symbols)
            if symbols:
                self.current_symbol = symbols[0]
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)
        controls_layout.addWidget(self.symbol_combo)
        
        # Timeframe selector
        controls_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        if hasattr(self.trading_system, 'config'):
            timeframes = self.trading_system.config.get("trading", {}).get("timeframes", [])
            self.timeframe_combo.addItems(timeframes)
            if timeframes:
                self.current_timeframe = timeframes[0]
        self.timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)
        controls_layout.addWidget(self.timeframe_combo)
        
        # Indicators
        controls_layout.addWidget(QLabel("Indicators:"))
        self.indicators_combo = QComboBox()
        self.indicators_combo.addItems(["None", "SMA", "EMA", "Bollinger Bands", "RSI", "MACD"])
        self.indicators_combo.currentTextChanged.connect(self._on_indicator_changed)
        controls_layout.addWidget(self.indicators_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        
        # Auto-refresh checkbox
        self.auto_refresh_cb = QCheckBox("Auto-refresh")
        self.auto_refresh_cb.toggled.connect(self._toggle_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_cb)
        
        layout.addLayout(controls_layout)
        
        # Chart widget
        self.chart_layout = QVBoxLayout()
        
        # Price plot
        self.price_plot = pg.PlotWidget()
        self.price_plot.setBackground('w')
        self.price_plot.showGrid(x=True, y=True)
        self.price_plot.setLabel('left', 'Price')
        
        # Volume plot
        self.volume_plot = pg.PlotWidget()
        self.volume_plot.setBackground('w')
        self.volume_plot.showGrid(x=True, y=True)
        self.volume_plot.setLabel('left', 'Volume')
        
        # Link x-axis of plots
        self.price_plot.setXLink(self.volume_plot)
        
        # Create layout for price and volume plots
        self.chart_layout.addWidget(self.price_plot, stretch=3)
        self.chart_layout.addWidget(self.volume_plot, stretch=1)
        
        layout.addLayout(self.chart_layout)
        
        # Initial data load
        self.refresh_data()
        
    def _on_symbol_changed(self, symbol):
        """Handle symbol change"""
        self.current_symbol = symbol
        self.refresh_data()
        
    def _on_timeframe_changed(self, timeframe):
        """Handle timeframe change"""
        self.current_timeframe = timeframe
        self.refresh_data()
        
    def _on_indicator_changed(self, indicator):
        """Handle indicator change"""
        if indicator != "None":
            if indicator not in self.indicators:
                self.indicators.append(indicator)
        else:
            self.indicators = []
            
        self.refresh_data()
        
    def _toggle_auto_refresh(self, enabled):
        """Toggle auto-refresh"""
        if enabled:
            self.refresh_timer.start(30000)  # Refresh every 30 seconds
        else:
            self.refresh_timer.stop()
        
    def refresh_data(self):
        """Refresh chart data"""
        if not self.current_symbol or not self.current_timeframe:
            return
            
        # Get market data
        data = self.trading_system.get_market_data(
            self.current_symbol, self.current_timeframe, limit=200)
            
        if data is None or data.empty:
            return
            
        self.current_data = data
        
        # Clear existing plots
        self.price_plot.clear()
        self.volume_plot.clear()
        
        # Plot candlestick chart
        self._plot_candlesticks(data)
        
        # Plot volume
        self._plot_volume(data)
        
        # Plot indicators
        for indicator in self.indicators:
            self._plot_indicator(indicator, data)
            
    def _plot_candlesticks(self, data):
        """Plot candlestick chart"""
        # Create time axis
        time_axis = np.arange(len(data))
        
        # Create CandlestickItem
        for i in range(len(data)):
            # Define candle colors
            if data['close'].iloc[i] > data['open'].iloc[i]:
                color = (0, 255, 0)  # Green for bullish
            else:
                color = (255, 0, 0)  # Red for bearish
                
            # Draw candle body
            body = pg.BarGraphItem(
                x=[i], 
                height=[data['close'].iloc[i] - data['open'].iloc[i]],
                width=0.8,
                brush=color,
                pen=color,
                y0=data['open'].iloc[i]
            )
            self.price_plot.addItem(body)
            
            # Draw wicks
            wick_pen = pg.mkPen(color=color, width=1)
            self.price_plot.plot(
                [i, i], 
                [data['low'].iloc[i], data['high'].iloc[i]], 
                pen=wick_pen
            )
        
    def _plot_volume(self, data):
        """Plot volume bars"""
        # Create time axis
        time_axis = np.arange(len(data))
        
        # Draw volume bars
        for i in range(len(data)):
            # Define bar colors
            if data['close'].iloc[i] > data['open'].iloc[i]:
                color = (0, 200, 0, 100)  # Green for bullish
            else:
                color = (200, 0, 0, 100)  # Red for bearish
                
            # Draw volume bar
            bar = pg.BarGraphItem(
                x=[i], 
                height=[data['volume'].iloc[i]],
                width=0.8,
                brush=color,
                pen=color
            )
            self.volume_plot.addItem(bar)
            
    def _plot_indicator(self, indicator, data):
        """Plot technical indicator"""
        if indicator == "SMA":
            # Plot SMA-20
            sma20 = data['close'].rolling(window=20).mean()
            self.price_plot.plot(np.arange(len(data)), sma20, pen=pg.mkPen('b', width=2))
            
            # Plot SMA-50
            sma50 = data['close'].rolling(window=50).mean()
            self.price_plot.plot(np.arange(len(data)), sma50, pen=pg.mkPen('r', width=2))
            
        elif indicator == "EMA":
            # Plot EMA-20
            ema20 = data['close'].ewm(span=20, adjust=False).mean()
            self.price_plot.plot(np.arange(len(data)), ema20, pen=pg.mkPen('b', width=2))
            
            # Plot EMA-50
            ema50 = data['close'].ewm(span=50, adjust=False).mean()
            self.price_plot.plot(np.arange(len(data)), ema50, pen=pg.mkPen('r', width=2))
            
        elif indicator == "Bollinger Bands":
            # Calculate Bollinger Bands
            sma20 = data['close'].rolling(window=20).mean()
            std20 = data['close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            # Plot middle band
            self.price_plot.plot(np.arange(len(data)), sma20, pen=pg.mkPen('b', width=2))
            
            # Plot upper and lower bands
            self.price_plot.plot(np.arange(len(data)), upper_band, pen=pg.mkPen('g', width=1))
            self.price_plot.plot(np.arange(len(data)), lower_band, pen=pg.mkPen('g', width=1))
            
        elif indicator == "RSI":
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Create RSI plot
            rsi_plot = pg.PlotWidget()
            rsi_plot.setBackground('w')
            rsi_plot.showGrid(x=True, y=True)
            rsi_plot.setLabel('left', 'RSI')
            rsi_plot.setXLink(self.price_plot)
            rsi_plot.setMaximumHeight(150)
            
            # Plot RSI
            rsi_plot.plot(np.arange(len(data)), rsi, pen=pg.mkPen('b', width=2))
            
            # Plot oversold/overbought lines
            rsi_plot.plot(np.arange(len(data)), [70] * len(data), pen=pg.mkPen('r', width=1, style=Qt.DashLine))
            rsi_plot.plot(np.arange(len(data)), [30] * len(data), pen=pg.mkPen('g', width=1, style=Qt.DashLine))
            
            # Add RSI plot to layout
            self.chart_layout.addWidget(rsi_plot)
            
        elif indicator == "MACD":
            # Calculate MACD
            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Create MACD plot
            macd_plot = pg.PlotWidget()
            macd_plot.setBackground('w')
            macd_plot.showGrid(x=True, y=True)
            macd_plot.setLabel('left', 'MACD')
            macd_plot.setXLink(self.price_plot)
            macd_plot.setMaximumHeight(150)
            
            # Plot MACD and signal lines
            macd_plot.plot(np.arange(len(data)), macd_line, pen=pg.mkPen('b', width=2))
            macd_plot.plot(np.arange(len(data)), signal_line, pen=pg.mkPen('r', width=2))
            
            # Plot histogram
            for i in range(len(histogram)):
                if i > 0:
                    if histogram.iloc[i] > 0:
                        color = (0, 200, 0, 150)  # Green
                    else:
                        color = (200, 0, 0, 150)  # Red
                        
                    bar = pg.BarGraphItem(
                        x=[i], 
                        height=[histogram.iloc[i]],
                        width=0.8,
                        brush=color,
                        pen=color
                    )
                    macd_plot.addItem(bar)
            
            # Add MACD plot to layout
            self.chart_layout.addWidget(macd_plot)
