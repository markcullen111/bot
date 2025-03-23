# gui/widgets/chart_widget.py

from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
    QPushButton, QFrame
)
from PyQt5.QtChart import (
    QChart, QChartView, QLineSeries, QValueAxis,
    QDateTimeAxis, QCandlestickSeries, QCandlestickSet
)
from PyQt5.QtCore import Qt, QDateTime, QPointF
from PyQt5.QtGui import QPainter
import pandas as pd
from datetime import datetime, timedelta

class ChartWidget(QWidget):
    """Widget for displaying price charts and technical indicators."""
    
    def __init__(self, trading_system):
        super().__init__()
        self.trading_system = trading_system
        self.current_symbol = None
        self.current_timeframe = "1h"
        self.indicators = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the chart UI."""
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Symbol selector
        self.symbol_selector = QComboBox()
        self.symbol_selector.addItems(
            self.trading_system.config['market_data']['symbols']
        )
        self.symbol_selector.currentTextChanged.connect(self.symbol_changed)
        controls_layout.addWidget(self.symbol_selector)
        
        # Timeframe selector
        self.timeframe_selector = QComboBox()
        self.timeframe_selector.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self.timeframe_selector.setCurrentText(self.current_timeframe)
        self.timeframe_selector.currentTextChanged.connect(self.timeframe_changed)
        controls_layout.addWidget(self.timeframe_selector)
        
        # Add indicators button
        add_indicator_btn = QPushButton("Add Indicator")
        add_indicator_btn.clicked.connect(self.add_indicator)
        controls_layout.addWidget(add_indicator_btn)
        
        # Add stretch to push everything to the left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Chart
        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # Create axes
        self.time_axis = QDateTimeAxis()
        self.time_axis.setFormat("yyyy-MM-dd hh:mm")
        self.time_axis.setTitleText("Time")
        self.chart.addAxis(self.time_axis, Qt.AlignBottom)
        
        self.price_axis = QValueAxis()
        self.price_axis.setTitleText("Price")
        self.chart.addAxis(self.price_axis, Qt.AlignLeft)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        
        # Add chart to layout
        layout.addWidget(self.chart_view)
        
        self.setLayout(layout)
        
        # Initialize with first symbol
        if self.symbol_selector.count() > 0:
            self.current_symbol = self.symbol_selector.currentText()
            self.update_chart()
    
    def symbol_changed(self, symbol: str):
        """Handle symbol change."""
        self.current_symbol = symbol
        self.update_chart()
    
    def timeframe_changed(self, timeframe: str):
        """Handle timeframe change."""
        self.current_timeframe = timeframe
        self.update_chart()
    
    def add_indicator(self):
        """Add a technical indicator to the chart."""
        # This would typically open a dialog to select and configure an indicator
        pass
    
    def update_chart(self):
        """Update the chart with new data."""
        if not self.current_symbol:
            return
        
        try:
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Last 30 days
            
            data = self.trading_system.broker.get_historical_data(
                self.current_symbol,
                self.current_timeframe,
                start_time.isoformat(),
                end_time.isoformat()
            )
            
            # Clear existing series
            self.chart.removeAllSeries()
            
            # Create candlestick series
            candle_series = QCandlestickSeries()
            candle_series.setName(self.current_symbol)
            
            # Add candlesticks
            prices = []
            for bar in data:
                timestamp = QDateTime.fromString(
                    bar['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    "yyyy-MM-dd hh:mm:ss"
                )
                
                candlestick = QCandlestickSet(
                    bar['open'],
                    bar['high'],
                    bar['low'],
                    bar['close'],
                    timestamp.toMSecsSinceEpoch()
                )
                candle_series.append(candlestick)
                prices.extend([bar['high'], bar['low']])
            
            self.chart.addSeries(candle_series)
            
            # Update axes
            min_price = min(prices)
            max_price = max(prices)
            price_margin = (max_price - min_price) * 0.1
            
            self.price_axis.setRange(
                min_price - price_margin,
                max_price + price_margin
            )
            
            start_msecs = QDateTime.fromString(
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "yyyy-MM-dd hh:mm:ss"
            ).toMSecsSinceEpoch()
            
            end_msecs = QDateTime.fromString(
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "yyyy-MM-dd hh:mm:ss"
            ).toMSecsSinceEpoch()
            
            self.time_axis.setRange(
                QDateTime.fromMSecsSinceEpoch(start_msecs),
                QDateTime.fromMSecsSinceEpoch(end_msecs)
            )
            
            # Attach axes
            candle_series.attachAxis(self.time_axis)
            candle_series.attachAxis(self.price_axis)
            
            # Update indicators
            self.update_indicators(data)
            
        except Exception as e:
            print(f"Error updating chart: {e}")
    
    def update_indicators(self, data: List[Dict]):
        """Update technical indicators on the chart."""
        # Example: Add a simple moving average
        try:
            df = pd.DataFrame(data)
            sma_period = 20
            sma = df['close'].rolling(window=sma_period).mean()
            
            sma_series = QLineSeries()
            sma_series.setName(f"SMA({sma_period})")
            
            for i, value in enumerate(sma):
                if pd.notna(value):
                    timestamp = QDateTime.fromString(
                        df['timestamp'].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
                        "yyyy-MM-dd hh:mm:ss"
                    )
                    sma_series.append(
                        QPointF(timestamp.toMSecsSinceEpoch(), value)
                    )
            
            self.chart.addSeries(sma_series)
            sma_series.attachAxis(self.time_axis)
            sma_series.attachAxis(self.price_axis)
            
        except Exception as e:
            print(f"Error updating indicators: {e}")
    
    def update_data(self):
        """Update chart with new data."""
        self.update_chart()
