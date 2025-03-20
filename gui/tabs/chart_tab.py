# chart_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QSplitter, QPushButton, QComboBox, 
                           QGroupBox, QCheckBox, QLineEdit, QListWidget, 
                           QListWidgetItem, QGridLayout, QFormLayout,
                           QColorDialog, QToolButton, QMenu, QAction,
                           QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush
import pyqtgraph as pg
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item for PyQtGraph"""
    
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # data must have fields: time, open, high, low, close, volume
        self.picture = None
        self.generatePicture()
        
    def generatePicture(self):
        """Pre-render the candlestick chart"""
        # Initialize QPicture object
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        
        w = 0.4  # Width of candlestick body
        for i in range(len(self.data['time'])):
            # Get OHLC data for this candle
            t = self.data['time'][i]
            open_val = self.data['open'][i]
            high_val = self.data['high'][i]
            low_val = self.data['low'][i]
            close_val = self.data['close'][i]
            
            # Determine color based on price movement
            if close_val > open_val:
                p.setPen(pg.mkPen('g'))
                p.setBrush(pg.mkBrush('g'))
            else:
                p.setPen(pg.mkPen('r'))
                p.setBrush(pg.mkBrush('r'))
                
            # Draw the candlestick body (rect from open to close)
            p.drawRect(pg.QtCore.QRectF(t - w, open_val, w * 2, close_val - open_val))
            
            # Draw the wicks (lines from low to high)
            p.drawLine(pg.QtCore.QPointF(t, low_val), pg.QtCore.QPointF(t, min(open_val, close_val)))
            p.drawLine(pg.QtCore.QPointF(t, max(open_val, close_val)), pg.QtCore.QPointF(t, high_val))
            
        p.end()
        
    def paint(self, p, *args):
        """Draw the cached candlestick picture"""
        p.drawPicture(0, 0, self.picture)
        
    def boundingRect(self):
        """Return the bounding rectangle of the candlestick chart"""
        return pg.QtCore.QRectF(self.picture.boundingRect())
        
    def update_data(self, data):
        """Update the candlestick data and redraw"""
        self.data = data
        self.generatePicture()
        self.update()

class ChartTab(QWidget):
    """Chart tab for displaying price charts and technical indicators"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.current_symbol = None
        self.current_timeframe = None
        self.candle_data = None
        self.active_indicators = {}  # Dictionary of active indicators
        self.indicator_plots = {}    # Store indicator plot items
        self.chart_annotations = []  # List of custom annotations (trendlines, etc.)
        
        # Initialize UI
        self._init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
        
        # Set default symbol and timeframe
        if self.trading_system.config["trading"]["symbols"]:
            self.symbol_combo.setCurrentText(self.trading_system.config["trading"]["symbols"][0])
            
        if self.trading_system.config["trading"]["timeframes"]:
            self.timeframe_combo.setCurrentText(self.trading_system.config["trading"]["timeframes"][0])
            
        # Initial data load
        self.load_chart_data()
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Top controls
        top_layout = QHBoxLayout()
        
        # Symbol selector
        top_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(self.trading_system.config["trading"]["symbols"])
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        top_layout.addWidget(self.symbol_combo)
        
        # Timeframe selector
        top_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(self.trading_system.config["trading"]["timeframes"])
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_changed)
        top_layout.addWidget(self.timeframe_combo)
        
        # Chart type selector
        top_layout.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Candlestick", "Line", "OHLC"])
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        top_layout.addWidget(self.chart_type_combo)
        
        # Data range selector
        top_layout.addWidget(QLabel("Range:"))
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "All"])
        self.range_combo.currentTextChanged.connect(self.on_range_changed)
        top_layout.addWidget(self.range_combo)
        
        # Add spacer
        top_layout.addStretch()
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        top_layout.addWidget(self.refresh_btn)
        
        main_layout.addLayout(top_layout)
        
        # Main content - split view
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side - chart and volume panel
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart with multiple plots for price and indicators
        self.chart_view = pg.GraphicsLayoutWidget()
        
        # Main price chart
        self.price_plot = self.chart_view.addPlot(row=0, col=0)
        self.price_plot.showGrid(x=True, y=True)
        self.price_plot.setLabel('left', 'Price')
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # Add mouse move event
        self.price_plot.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Create volume plot linked with price chart x-axis
        self.volume_plot = self.chart_view.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.setMaximumHeight(100)
        
        # Link X axes for synchronized zooming and panning
        self.volume_plot.setXLink(self.price_plot)
        
        chart_layout.addWidget(self.chart_view)
        
        self.splitter.addWidget(chart_widget)
        
        # Right side - indicator controls
        indicator_widget = QWidget()
        indicator_layout = QVBoxLayout(indicator_widget)
        
        # Indicators group
        indicators_group = QGroupBox("Technical Indicators")
        indicators_form = QFormLayout(indicators_group)
        
        # Add indicator button
        self.add_indicator_combo = QComboBox()
        self.add_indicator_combo.addItems([
            "Moving Average", "Bollinger Bands", "RSI", 
            "MACD", "Stochastic", "ATR", "ADX", "Ichimoku"
        ])
        
        self.add_indicator_btn = QPushButton("Add")
        self.add_indicator_btn.clicked.connect(self.add_indicator)
        
        add_layout = QHBoxLayout()
        add_layout.addWidget(self.add_indicator_combo)
        add_layout.addWidget(self.add_indicator_btn)
        indicators_form.addRow("Add Indicator:", add_layout)
        
        # Active indicators list
        self.indicators_list = QListWidget()
        self.indicators_list.setMinimumHeight(200)
        self.indicators_list.itemDoubleClicked.connect(self.edit_indicator)
        self.indicators_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.indicators_list.customContextMenuRequested.connect(self.show_indicator_context_menu)
        indicators_form.addRow("Active Indicators:", self.indicators_list)
        
        indicator_layout.addWidget(indicators_group)
        
        # Drawing tools group
        drawing_group = QGroupBox("Chart Annotations")
        drawing_layout = QVBoxLayout(drawing_group)
        
        # Drawing tools
        tools_layout = QHBoxLayout()
        
        self.trendline_btn = QToolButton()
        self.trendline_btn.setText("Trendline")
        self.trendline_btn.clicked.connect(lambda: self.activate_drawing_tool("trendline"))
        tools_layout.addWidget(self.trendline_btn)
        
        self.horizontal_btn = QToolButton()
        self.horizontal_btn.setText("Horizontal")
        self.horizontal_btn.clicked.connect(lambda: self.activate_drawing_tool("horizontal"))
        tools_layout.addWidget(self.horizontal_btn)
        
        self.vertical_btn = QToolButton()
        self.vertical_btn.setText("Vertical")
        self.vertical_btn.clicked.connect(lambda: self.activate_drawing_tool("vertical"))
        tools_layout.addWidget(self.vertical_btn)
        
        self.rect_btn = QToolButton()
        self.rect_btn.setText("Rectangle")
        self.rect_btn.clicked.connect(lambda: self.activate_drawing_tool("rectangle"))
        tools_layout.addWidget(self.rect_btn)
        
        self.text_btn = QToolButton()
        self.text_btn.setText("Text")
        self.text_btn.clicked.connect(lambda: self.activate_drawing_tool("text"))
        tools_layout.addWidget(self.text_btn)
        
        self.fib_btn = QToolButton()
        self.fib_btn.setText("Fibonacci")
        self.fib_btn.clicked.connect(lambda: self.activate_drawing_tool("fibonacci"))
        tools_layout.addWidget(self.fib_btn)
        
        drawing_layout.addLayout(tools_layout)
        
        # Drawing color picker
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        
        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet("background-color: #FF0000;")
        self.color_btn.setFixedSize(24, 24)
        self.color_btn.clicked.connect(self.select_drawing_color)
        color_layout.addWidget(self.color_btn)
        
        # Clear annotations button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_annotations)
        color_layout.addWidget(self.clear_btn)
        
        drawing_layout.addLayout(color_layout)
        
        indicator_layout.addWidget(drawing_group)
        
        # Chart settings
        settings_group = QGroupBox("Chart Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Grid settings
        grid_layout = QHBoxLayout()
        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.stateChanged.connect(self.update_chart_settings)
        grid_layout.addWidget(self.show_grid_cb)
        settings_layout.addLayout(grid_layout)
        
        # Theme settings
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.set_chart_theme)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addLayout(theme_layout)
        
        # Export chart button
        self.export_btn = QPushButton("Export Chart")
        self.export_btn.clicked.connect(self.export_chart)
        settings_layout.addWidget(self.export_btn)
        
        indicator_layout.addWidget(settings_group)
        
        # Add stretch to bottom
        indicator_layout.addStretch()
        
        # Set fixed width for the controls panel
        indicator_widget.setFixedWidth(300)
        
        self.splitter.addWidget(indicator_widget)
        
        main_layout.addWidget(self.splitter)
        
        # Set initial chart theme
        self.set_chart_theme("Light")
        
    def load_chart_data(self):
        """Load chart data for the selected symbol and timeframe"""
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        
        if not symbol or not timeframe:
            return
            
        # Store current selections
        self.current_symbol = symbol
        self.current_timeframe = timeframe
        
        # Calculate date range based on selection
        limit = self.get_limit_from_range()
        
        # Get market data
        data = self.trading_system.get_market_data(symbol, timeframe, limit=limit)
        
        if data.empty:
            return
            
        # Store data
        self.candle_data = data
        
        # Update chart
        self.update_chart()
        
    def update_chart(self):
        """Update chart with current data"""
        if self.candle_data is None or self.candle_data.empty:
            return
            
        # Clear existing items
        self.price_plot.clear()
        self.volume_plot.clear()
        
        # Reset indicator plots
        self.indicator_plots = {}
        
        # Get chart type
        chart_type = self.chart_type_combo.currentText()
        
        # Prepare data
        data = self.candle_data.copy()
        
        # Convert index to timestamp
        timestamps = np.array(range(len(data.index)))
        
        # Create dict for candlestick data
        ohlc_data = {
            'time': timestamps,
            'open': data['open'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'close': data['close'].values,
            'volume': data['volume'].values if 'volume' in data else np.zeros(len(data))
        }
        
        # Display chart based on selected type
        if chart_type == "Candlestick":
            # Create candlestick item
            candlestick = CandlestickItem(ohlc_data)
            self.price_plot.addItem(candlestick)
        elif chart_type == "Line":
            # Create line chart
            self.price_plot.plot(timestamps, data['close'].values, pen='b')
        elif chart_type == "OHLC":
            # Simplified OHLC - just show candlesticks for now
            # In a real implementation, we would have a proper OHLC renderer
            candlestick = CandlestickItem(ohlc_data)
            self.price_plot.addItem(candlestick)
            
        # Add volume bars
        if 'volume' in data:
            volume_data = data['volume'].values
            volume_bars = pg.BarGraphItem(x=timestamps, height=volume_data, width=0.6, brush='b')
            self.volume_plot.addItem(volume_bars)
            
        # Set X axis range
        self.price_plot.setXRange(timestamps[0], timestamps[-1])
        
        # Add time axis labels with date ticks
        class DateAxisItem(pg.AxisItem):
            def tickStrings(self, values, scale, spacing):
                strings = []
                for v in values:
                    try:
                        string = data.index[int(v)].strftime('%m-%d %H:%M')
                    except (ValueError, IndexError):
                        string = ''
                    strings.append(string)
                return strings
                
        date_axis = DateAxisItem(orientation='bottom')
        self.price_plot.setAxisItems({'bottom': date_axis})
        
        # Update indicators
        self.update_indicators()
        
    def update_indicators(self):
        """Update technical indicators on the chart"""
        if self.candle_data is None or self.candle_data.empty:
            return
            
        data = self.candle_data.copy()
        timestamps = np.array(range(len(data.index)))
        
        # Process each active indicator
        for indicator_id, indicator_config in self.active_indicators.items():
            indicator_type = indicator_config['type']
            params = indicator_config.get('params', {})
            
            # Get the values for the indicator
            if indicator_type == "Moving Average":
                period = params.get('period', 20)
                ma_type = params.get('ma_type', 'SMA')
                
                if ma_type == 'SMA':
                    values = data['close'].rolling(window=period).mean()
                else:  # EMA
                    values = data['close'].ewm(span=period, adjust=False).mean()
                    
                # Plot on price chart
                color = params.get('color', 'blue')
                self.indicator_plots[indicator_id] = self.price_plot.plot(
                    timestamps, values.values, pen=pg.mkPen(color, width=1.5), name=f"{ma_type}({period})"
                )
                
            elif indicator_type == "Bollinger Bands":
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2)
                
                # Calculate BB
                middle = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
                # Plot bands
                self.indicator_plots[f"{indicator_id}_middle"] = self.price_plot.plot(
                    timestamps, middle.values, pen=pg.mkPen('blue', width=1.5), name=f"BB Middle({period})"
                )
                self.indicator_plots[f"{indicator_id}_upper"] = self.price_plot.plot(
                    timestamps, upper.values, pen=pg.mkPen('blue', width=1, style=Qt.DashLine), name=f"BB Upper({period})"
                )
                self.indicator_plots[f"{indicator_id}_lower"] = self.price_plot.plot(
                    timestamps, lower.values, pen=pg.mkPen('blue', width=1, style=Qt.DashLine), name=f"BB Lower({period})"
                )
                
            elif indicator_type == "RSI":
                period = params.get('period', 14)
                
                # Calculate RSI
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Create sub-plot for RSI
                rsi_plot = self.chart_view.addPlot(row=2, col=0)
                rsi_plot.setXLink(self.price_plot)
                rsi_plot.setMaximumHeight(100)
                rsi_plot.setLabel('left', 'RSI')
                
                # Add RSI line
                self.indicator_plots[indicator_id] = rsi_plot.plot(
                    timestamps, rsi.values, pen=pg.mkPen('purple', width=1.5), name=f"RSI({period})"
                )
                
                # Add reference lines
                rsi_plot.addLine(y=70, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
                rsi_plot.addLine(y=30, pen=pg.mkPen('g', width=1, style=Qt.DashLine))
                
            elif indicator_type == "MACD":
                fast_period = params.get('fast_period', 12)
                slow_period = params.get('slow_period', 26)
                signal_period = params.get('signal_period', 9)
                
                # Calculate MACD
                ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
                ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                histogram = macd_line - signal_line
                
                # Create sub-plot for MACD
                macd_plot = self.chart_view.addPlot(row=3, col=0)
                macd_plot.setXLink(self.price_plot)
                macd_plot.setMaximumHeight(100)
                macd_plot.setLabel('left', 'MACD')
                
                # Add MACD lines
                self.indicator_plots[f"{indicator_id}_macd"] = macd_plot.plot(
                    timestamps, macd_line.values, pen=pg.mkPen('blue', width=1.5), name="MACD"
                )
                self.indicator_plots[f"{indicator_id}_signal"] = macd_plot.plot(
                    timestamps, signal_line.values, pen=pg.mkPen('red', width=1.5), name="Signal"
                )
                
                # Add histogram as bar graph
                bar_item = pg.BarGraphItem(x=timestamps, height=histogram.values, width=0.6, brush='g')
                macd_plot.addItem(bar_item)
                self.indicator_plots[f"{indicator_id}_histogram"] = bar_item
                
            # Add more indicators as needed...
            
    def on_symbol_changed(self, symbol):
        """Handle symbol selection change"""
        self.load_chart_data()
        
    def on_timeframe_changed(self, timeframe):
        """Handle timeframe selection change"""
        self.load_chart_data()
        
    def on_chart_type_changed(self, chart_type):
        """Handle chart type selection change"""
        self.update_chart()
        
    def on_range_changed(self, range_text):
        """Handle range selection change"""
        self.load_chart_data()
        
    def get_limit_from_range(self):
        """Convert range selection to a data limit"""
        range_text = self.range_combo.currentText()
        
        if range_text == "1 Day":
            return 24  # Assuming hourly data
        elif range_text == "1 Week":
            return 168  # 7 * 24 hours
        elif range_text == "1 Month":
            return 720  # ~30 * 24 hours
        elif range_text == "3 Months":
            return 2160  # ~90 * 24 hours
        elif range_text == "6 Months":
            return 4320  # ~180 * 24 hours
        elif range_text == "1 Year":
            return 8760  # ~365 * 24 hours
        else:  # All
            return 10000  # Large number to get all available data
            
    def refresh_data(self):
        """Refresh chart data"""
        self.load_chart_data()
        
    def add_indicator(self):
        """Add a new technical indicator to the chart"""
        indicator_type = self.add_indicator_combo.currentText()
        
        # Create unique ID for the indicator
        indicator_id = f"{indicator_type}_{len(self.active_indicators)}"
        
        # Default parameters for each indicator type
        default_params = {
            "Moving Average": {"period": 20, "ma_type": "SMA", "color": "blue"},
            "Bollinger Bands": {"period": 20, "std_dev": 2},
            "RSI": {"period": 14},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "Stochastic": {"k_period": 14, "d_period": 3},
            "ATR": {"period": 14},
            "ADX": {"period": 14},
            "Ichimoku": {}
        }
        
        # Add to active indicators
        self.active_indicators[indicator_id] = {
            "type": indicator_type,
            "params": default_params.get(indicator_type, {})
        }
        
        # Add to list widget
        item = QListWidgetItem(f"{indicator_type}")
        item.setData(Qt.UserRole, indicator_id)
        self.indicators_list.addItem(item)
        
        # Update chart
        self.update_chart()
        
    def edit_indicator(self, item):
        """Edit an existing indicator"""
        indicator_id = item.data(Qt.UserRole)
        if indicator_id not in self.active_indicators:
            return
            
        indicator_config = self.active_indicators[indicator_id]
        indicator_type = indicator_config['type']
        params = indicator_config.get('params', {})
        
        # Show edit dialog based on indicator type
        if indicator_type == "Moving Average":
            period, ok1 = QInputDialog.getInt(
                self, "Edit Moving Average", "Period:", 
                params.get('period', 20), 2, 200
            )
            if not ok1:
                return
                
            ma_types = ["SMA", "EMA"]
            ma_type, ok2 = QInputDialog.getItem(
                self, "Edit Moving Average", "Type:",
                ma_types, ma_types.index(params.get('ma_type', 'SMA')), False
            )
            if not ok2:
                return
                
            # Update parameters
            self.active_indicators[indicator_id]['params']['period'] = period
            self.active_indicators[indicator_id]['params']['ma_type'] = ma_type
            
            # Update item text
            item.setText(f"{indicator_type} - {ma_type}({period})")
            
        elif indicator_type == "Bollinger Bands":
            period, ok1 = QInputDialog.getInt(
                self, "Edit Bollinger Bands", "Period:", 
                params.get('period', 20), 2, 200
            )
            if not ok1:
                return
                
            std_dev, ok2 = QInputDialog.getDouble(
                self, "Edit Bollinger Bands", "Standard Deviations:",
                params.get('std_dev', 2), 0.1, 5, 1
            )
            if not ok2:
                return
                
            # Update parameters
            self.active_indicators[indicator_id]['params']['period'] = period
            self.active_indicators[indicator_id]['params']['std_dev'] = std_dev
            
            # Update item text
            item.setText(f"{indicator_type} ({period}, {std_dev})")
            
        elif indicator_type == "RSI":
            period, ok = QInputDialog.getInt(
                self, "Edit RSI", "Period:", 
                params.get('period', 14), 2, 200
            )
            if not ok:
                return
                
            # Update parameters
            self.active_indicators[indicator_id]['params']['period'] = period
            
            # Update item text
            item.setText(f"{indicator_type} ({period})")
            
        elif indicator_type == "MACD":
            fast_period, ok1 = QInputDialog.getInt(
                self, "Edit MACD", "Fast Period:", 
                params.get('fast_period', 12), 2, 200
            )
            if not ok1:
                return
                
            slow_period, ok2 = QInputDialog.getInt(
                self, "Edit MACD", "Slow Period:", 
                params.get('slow_period', 26), 2, 200
            )
            if not ok2:
                return
                
            signal_period, ok3 = QInputDialog.getInt(
                self, "Edit MACD", "Signal Period:", 
                params.get('signal_period', 9), 2, 200
            )
            if not ok3:
                return
                
            # Update parameters
            self.active_indicators[indicator_id]['params']['fast_period'] = fast_period
            self.active_indicators[indicator_id]['params']['slow_period'] = slow_period
            self.active_indicators[indicator_id]['params']['signal_period'] = signal_period
            
            # Update item text
            item.setText(f"{indicator_type} ({fast_period}, {slow_period}, {signal_period})")
            
        # Add more indicator types as needed...
        
        # Update chart
        self.update_chart()
        
    def show_indicator_context_menu(self, position):
        """Show context menu for indicators list"""
        if self.indicators_list.count() == 0:
            return
            
        item = self.indicators_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        edit_action = menu.addAction("Edit")
        remove_action = menu.addAction("Remove")
        
        action = menu.exec_(self.indicators_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_indicator(item)
        elif action == remove_action:
            indicator_id = item.data(Qt.UserRole)
            if indicator_id in self.active_indicators:
                del self.active_indicators[indicator_id]
                self.indicators_list.takeItem(self.indicators_list.row(item))
                self.update_chart()
                
    def mouse_moved(self, pos):
        """Handle mouse movement over chart for crosshair"""
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())
            
    def activate_drawing_tool(self, tool_type):
        """Activate a drawing tool"""
        # This would implement drawing tool functionality
        # In a real implementation, this would set up event handlers
        # for mouse events to draw the selected annotation
        pass
        
    def select_drawing_color(self):
        """Select color for drawing tools"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")
            
    def clear_annotations(self):
        """Clear all chart annotations"""
        # This would remove all user-drawn annotations
        # In a real implementation, we would track annotation objects
        # and remove them from the plot
        pass
        
    def update_chart_settings(self):
        """Update chart display settings"""
        # Update grid
        show_grid = self.show_grid_cb.isChecked()
        self.price_plot.showGrid(x=show_grid, y=show_grid)
        self.volume_plot.showGrid(x=show_grid, y=show_grid)
        
    def set_chart_theme(self, theme):
        """Set chart color theme"""
        if theme == "Light":
            # Light theme
            self.chart_view.setBackground('w')
            self.price_plot.getAxis('bottom').setPen('k')
            self.price_plot.getAxis('left').setPen('k')
            self.volume_plot.getAxis('bottom').setPen('k')
            self.volume_plot.getAxis('left').setPen('k')
        else:
            # Dark theme
            self.chart_view.setBackground('k')
            self.price_plot.getAxis('bottom').setPen('w')
            self.price_plot.getAxis('left').setPen('w')
            self.volume_plot.getAxis('bottom').setPen('w')
            self.volume_plot.getAxis('left').setPen('w')
            
        # Update chart
        self.update_chart()
        
    def export_chart(self):
        """Export current chart as image"""
        # This would implement chart export functionality
        # In a real implementation, we would use something like:
        # exporter = pg.exporters.ImageExporter(self.chart_view.scene())
        # exporter.export('chart_export.png')
        
        QMessageBox.information(self, "Export Chart", "Chart export functionality would save the current chart as an image.")