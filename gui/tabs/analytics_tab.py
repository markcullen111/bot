# gui/tabs/analytics_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QComboBox, QPushButton, QTabWidget,
                           QSplitter, QGridLayout, QGroupBox, QTableWidget,
                           QTableWidgetItem, QHeaderView, QDateEdit,
                           QFormLayout, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class CorrelationMatrixWidget(QWidget):
    """Widget for displaying correlation matrix"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Symbols:"))
        self.symbols_combo = QComboBox()
        self.symbols_combo.addItems(self.trading_system.config["trading"]["symbols"])
        self.symbols_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.symbols_combo)
        
        controls_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(self.trading_system.config["trading"]["timeframes"])
        self.timeframe_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.timeframe_combo)
        
        controls_layout.addWidget(QLabel("Days:"))
        self.days_spin = QSpinBox()
        self.days_spin.setMinimum(1)
        self.days_spin.setMaximum(365)
        self.days_spin.setValue(30)
        controls_layout.addWidget(self.days_spin)
        
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self.update_correlation_matrix)
        controls_layout.addWidget(self.calculate_btn)
        
        layout.addLayout(controls_layout)
        
        # Matplotlib figure for correlation matrix heatmap
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
    def update_correlation_matrix(self):
        """Update correlation matrix visualization"""
        try:
            symbol = self.symbols_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            days = self.days_spin.value()
            
            # Get data
            start_date = datetime.now() - timedelta(days=days)
            data = self.trading_system.get_market_data(
                symbol, timeframe, start_time=start_date
            )
            
            if data.empty:
                print("No data available for correlation matrix")
                return
                
            # Calculate correlation matrix
            price_columns = ['open', 'high', 'low', 'close']
            corr_matrix = data[price_columns].corr()
            
            # Create heatmap
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            self.figure.colorbar(cax)
            
            # Set labels
            ax.set_xticks(np.arange(len(price_columns)))
            ax.set_yticks(np.arange(len(price_columns)))
            ax.set_xticklabels(price_columns)
            ax.set_yticklabels(price_columns)
            
            # Add correlation values
            for i in range(len(price_columns)):
                for j in range(len(price_columns)):
                    ax.text(i, j, f"{corr_matrix.iloc[j, i]:.2f}",
                           ha="center", va="center", color="black")
                           
            # Set title
            ax.set_title(f"Price Correlation Matrix - {symbol} ({timeframe})")
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating correlation matrix: {e}")

class ModelPerformanceWidget(QWidget):
    """Widget for tracking model prediction performance"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        # Add model types - these would be populated based on your system
        self.model_combo.addItems([
            "Order Flow Prediction", 
            "Trade Timing", 
            "Trade Exit", 
            "Trade Reentry", 
            "ML Ensemble"
        ])
        self.model_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.model_combo)
        
        controls_layout.addWidget(QLabel("Time Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])
        self.period_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.period_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_model_performance)
        controls_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(controls_layout)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        metrics_layout.addWidget(QLabel("Accuracy:"), 0, 0)
        self.accuracy_label = QLabel("0.00%")
        metrics_layout.addWidget(self.accuracy_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Precision:"), 1, 0)
        self.precision_label = QLabel("0.00%")
        metrics_layout.addWidget(self.precision_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Recall:"), 2, 0)
        self.recall_label = QLabel("0.00%")
        metrics_layout.addWidget(self.recall_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("F1 Score:"), 3, 0)
        self.f1_label = QLabel("0.00")
        metrics_layout.addWidget(self.f1_label, 3, 1)
        
        metrics_layout.addWidget(QLabel("RMSE:"), 0, 2)
        self.rmse_label = QLabel("0.00")
        metrics_layout.addWidget(self.rmse_label, 0, 3)
        
        metrics_layout.addWidget(QLabel("MAE:"), 1, 2)
        self.mae_label = QLabel("0.00")
        metrics_layout.addWidget(self.mae_label, 1, 3)
        
        metrics_layout.addWidget(QLabel("R²:"), 2, 2)
        self.r2_label = QLabel("0.00")
        metrics_layout.addWidget(self.r2_label, 2, 3)
        
        metrics_layout.addWidget(QLabel("Sharpe Impact:"), 3, 2)
        self.sharpe_impact_label = QLabel("0.00")
        metrics_layout.addWidget(self.sharpe_impact_label, 3, 3)
        
        layout.addWidget(metrics_group)
        
        # Performance chart
        chart_layout = QVBoxLayout()
        chart_layout.addWidget(QLabel("Prediction Accuracy Over Time"))
        
        self.performance_chart = pg.PlotWidget()
        self.performance_chart.setBackground('w')
        self.performance_chart.showGrid(x=True, y=True)
        self.performance_chart.setLabel('left', 'Accuracy')
        self.performance_chart.setLabel('bottom', 'Date')
        
        chart_layout.addWidget(self.performance_chart)
        layout.addLayout(chart_layout)
        
        # Prediction vs Actual
        comparison_layout = QVBoxLayout()
        comparison_layout.addWidget(QLabel("Prediction vs Actual"))
        
        self.comparison_chart = pg.PlotWidget()
        self.comparison_chart.setBackground('w')
        self.comparison_chart.showGrid(x=True, y=True)
        self.comparison_chart.setLabel('left', 'Value')
        self.comparison_chart.setLabel('bottom', 'Time')
        
        comparison_layout.addWidget(self.comparison_chart)
        layout.addLayout(comparison_layout)
        
        # Initial update
        self.update_model_performance()
        
    def update_model_performance(self):
        """Update model performance visualizations"""
        try:
            model_name = self.model_combo.currentText()
            period = self.period_combo.currentText()
            
            # Get time range
            end_date = datetime.now()
            if period == "1 Week":
                start_date = end_date - timedelta(days=7)
            elif period == "1 Month":
                start_date = end_date - timedelta(days=30)
            elif period == "3 Months":
                start_date = end_date - timedelta(days=90)
            elif period == "6 Months":
                start_date = end_date - timedelta(days=180)
            else:  # 1 Year
                start_date = end_date - timedelta(days=365)
                
            # In a real implementation, you would fetch actual model performance data
            # For demonstration, we'll generate some random data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate realistic performance metrics
            accuracy = 0.75 + np.random.normal(0, 0.05)
            precision = 0.70 + np.random.normal(0, 0.05)
            recall = 0.72 + np.random.normal(0, 0.05)
            f1 = 2 * (precision * recall) / (precision + recall)
            rmse = 0.12 + np.random.normal(0, 0.02)
            mae = 0.09 + np.random.normal(0, 0.015)
            r2 = 0.65 + np.random.normal(0, 0.1)
            sharpe_impact = 0.35 + np.random.normal(0, 0.1)
            
            # Update metrics labels
            self.accuracy_label.setText(f"{accuracy * 100:.2f}%")
            self.precision_label.setText(f"{precision * 100:.2f}%")
            self.recall_label.setText(f"{recall * 100:.2f}%")
            self.f1_label.setText(f"{f1:.2f}")
            self.rmse_label.setText(f"{rmse:.4f}")
            self.mae_label.setText(f"{mae:.4f}")
            self.r2_label.setText(f"{r2:.2f}")
            self.sharpe_impact_label.setText(f"{sharpe_impact:.2f}")
            
            # Generate performance over time data
            accuracies = []
            for _ in range(len(dates)):
                accuracies.append(0.7 + np.random.normal(0, 0.08))
                
            # Update performance chart
            self.performance_chart.clear()
            pen = pg.mkPen(color='b', width=2)
            self.performance_chart.plot(range(len(dates)), accuracies, pen=pen)
            
            # Generate prediction vs actual data
            predictions = []
            actuals = []
            
            # Start with some base value
            base = 100
            
            for _ in range(30):  # Last 30 days
                # Generate actual value with some trend and randomness
                actual = base + np.random.normal(0, 2)
                actuals.append(actual)
                
                # Generate prediction with some error
                prediction = actual + np.random.normal(0, 3)
                predictions.append(prediction)
                
                # Update base for next point
                base = actual + np.random.normal(0.1, 0.5)  # Slight upward trend
                
            # Update comparison chart
            self.comparison_chart.clear()
            
            # Plot actuals
            pen_actual = pg.mkPen(color='b', width=2)
            self.comparison_chart.plot(range(len(actuals)), actuals, pen=pen_actual, name="Actual")
            
            # Plot predictions
            pen_pred = pg.mkPen(color='r', width=2)
            self.comparison_chart.plot(range(len(predictions)), predictions, pen=pen_pred, name="Predicted")
            
            # Add legend
            self.comparison_chart.addLegend()
            
        except Exception as e:
            print(f"Error updating model performance: {e}")

class PortfolioAnalysisWidget(QWidget):
    """Widget for portfolio analysis and allocation visualization"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Time Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 Week", "1 Month", "3 Months", "All Time"])
        self.period_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.period_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_portfolio_analysis)
        controls_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(controls_layout)
        
        # Split views
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - allocation pie chart
        allocation_widget = QWidget()
        allocation_layout = QVBoxLayout(allocation_widget)
        
        allocation_layout.addWidget(QLabel("Current Portfolio Allocation"))
        
        self.allocation_figure = Figure(figsize=(6, 5))
        self.allocation_canvas = FigureCanvas(self.allocation_figure)
        allocation_layout.addWidget(self.allocation_canvas)
        
        splitter.addWidget(allocation_widget)
        
        # Right side - performance metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Performance table
        metrics_layout.addWidget(QLabel("Asset Performance"))
        
        self.performance_table = QTableWidget(0, 5)
        self.performance_table.setHorizontalHeaderLabels([
            "Asset", "Allocation", "Return", "Contribution", "Sharpe"
        ])
        self.performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.performance_table.verticalHeader().setVisible(False)
        
        metrics_layout.addWidget(self.performance_table)
        
        # Correlation heatmap
        metrics_layout.addWidget(QLabel("Asset Correlation"))
        
        self.correlation_figure = Figure(figsize=(6, 4))
        self.correlation_canvas = FigureCanvas(self.correlation_figure)
        metrics_layout.addWidget(self.correlation_canvas)
        
        splitter.addWidget(metrics_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        # Initial update
        self.update_portfolio_analysis()
        
    def update_portfolio_analysis(self):
        """Update portfolio analysis visualizations"""
        try:
            # In a real implementation, you would fetch actual portfolio data
            
            # Get open positions
            positions = self.trading_system.open_positions
            
            # If no positions, show placeholder data
            if not positions:
                # Generate some sample data
                assets = ["BTC", "ETH", "SOL", "ADA", "DOT"]
                allocations = [0.4, 0.3, 0.15, 0.1, 0.05]
                returns = [0.12, 0.08, 0.15, -0.05, 0.10]
                sharpes = [1.2, 1.0, 1.3, 0.8, 1.1]
            else:
                # Extract data from actual positions
                assets = []
                allocations = []
                returns = []
                sharpes = []
                
                portfolio_value = sum(p.get('current_value', 0) for p in positions.values())
                
                for symbol, position in positions.items():
                    assets.append(symbol.split('/')[0])
                    
                    # Calculate allocation
                    position_value = position.get('current_value', 0)
                    allocation = position_value / portfolio_value if portfolio_value > 0 else 0
                    allocations.append(allocation)
                    
                    # For return and sharpe, we'd need historical data
                    # Using placeholders for demonstration
                    returns.append(0.1)  # 10% return
                    sharpes.append(1.1)  # Sharpe ratio
            
            # Update allocation pie chart
            self.allocation_figure.clear()
            ax = self.allocation_figure.add_subplot(111)
            
            ax.pie(allocations, labels=assets, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Portfolio Allocation')
            
            self.allocation_canvas.draw()
            
            # Update performance table
            self.performance_table.setRowCount(len(assets))
            
            for i, (asset, allocation, ret, sharpe) in enumerate(zip(assets, allocations, returns, sharpes)):
                # Calculate contribution to portfolio return
                contribution = allocation * ret
                
                self.performance_table.setItem(i, 0, QTableWidgetItem(asset))
                self.performance_table.setItem(i, 1, QTableWidgetItem(f"{allocation * 100:.1f}%"))
                
                # Color code returns
                return_item = QTableWidgetItem(f"{ret * 100:.2f}%")
                if ret > 0:
                    return_item.setForeground(QColor('green'))
                elif ret < 0:
                    return_item.setForeground(QColor('red'))
                self.performance_table.setItem(i, 2, return_item)
                
                # Contribution
                contribution_item = QTableWidgetItem(f"{contribution * 100:.2f}%")
                if contribution > 0:
                    contribution_item.setForeground(QColor('green'))
                elif contribution < 0:
                    contribution_item.setForeground(QColor('red'))
                self.performance_table.setItem(i, 3, contribution_item)
                
                self.performance_table.setItem(i, 4, QTableWidgetItem(f"{sharpe:.2f}"))
            
            # Update correlation heatmap
            self.correlation_figure.clear()
            ax = self.correlation_figure.add_subplot(111)
            
            # Generate sample correlation matrix
            num_assets = len(assets)
            corr_matrix = np.eye(num_assets)  # Start with identity matrix
            
            # Fill with realistic correlations
            for i in range(num_assets):
                for j in range(i+1, num_assets):
                    # Generate correlation between 0.3 and 0.9
                    correlation = 0.3 + np.random.random() * 0.6
                    corr_matrix[i, j] = correlation
                    corr_matrix[j, i] = correlation
            
            cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            self.correlation_figure.colorbar(cax)
            
            # Set labels
            ax.set_xticks(np.arange(len(assets)))
            ax.set_yticks(np.arange(len(assets)))
            ax.set_xticklabels(assets)
            ax.set_yticklabels(assets)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add correlation values
            for i in range(len(assets)):
                for j in range(len(assets)):
                    ax.text(i, j, f"{corr_matrix[j, i]:.2f}",
                           ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title("Asset Correlation Matrix")
            
            self.correlation_canvas.draw()
            
        except Exception as e:
            print(f"Error updating portfolio analysis: {e}")

class MarketRegimeWidget(QWidget):
    """Widget for analyzing market regimes and conditions"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(self.trading_system.config["trading"]["symbols"])
        self.symbol_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.symbol_combo)
        
        controls_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(self.trading_system.config["trading"]["timeframes"])
        self.timeframe_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.timeframe_combo)
        
        controls_layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 Month", "3 Months", "6 Months", "1 Year"])
        self.period_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.period_combo)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_market_regime)
        controls_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(controls_layout)
        
        # Market regime visualization
        regime_layout = QHBoxLayout()
        
        # Current state
        current_group = QGroupBox("Current Market State")
        current_layout = QFormLayout(current_group)
        
        current_layout.addRow("Current Regime:", QLabel("Unknown"))
        self.regime_label = current_layout.itemAt(0, QFormLayout.FieldRole).widget()
        
        current_layout.addRow("Trend Strength:", QLabel("0.0"))
        self.trend_label = current_layout.itemAt(1, QFormLayout.FieldRole).widget()
        
        current_layout.addRow("Volatility:", QLabel("Low"))
        self.volatility_label = current_layout.itemAt(2, QFormLayout.FieldRole).widget()
        
        current_layout.addRow("Momentum:", QLabel("Neutral"))
        self.momentum_label = current_layout.itemAt(3, QFormLayout.FieldRole).widget()
        
        current_layout.addRow("Support Level:", QLabel("$0.00"))
        self.support_label = current_layout.itemAt(4, QFormLayout.FieldRole).widget()
        
        current_layout.addRow("Resistance Level:", QLabel("$0.00"))
        self.resistance_label = current_layout.itemAt(5, QFormLayout.FieldRole).widget()
        
        regime_layout.addWidget(current_group)
        
        # Regime histogram
        histogram_widget = QWidget()
        histogram_layout = QVBoxLayout(histogram_widget)
        
        histogram_layout.addWidget(QLabel("Market Regime Distribution"))
        
        self.regime_figure = Figure(figsize=(8, 4))
        self.regime_canvas = FigureCanvas(self.regime_figure)
        histogram_layout.addWidget(self.regime_canvas)
        
        regime_layout.addWidget(histogram_widget)
        
        layout.addLayout(regime_layout)
        
        # Regime transitions visualization
        layout.addWidget(QLabel("Market Regime Transitions"))
        
        self.transitions_chart = pg.PlotWidget()
        self.transitions_chart.setBackground('w')
        self.transitions_chart.showGrid(x=True, y=True)
        
        layout.addWidget(self.transitions_chart)
        
        # Initial analysis
        self.analyze_market_regime()
        
    def analyze_market_regime(self):
        """Perform market regime analysis"""
        try:
            symbol = self.symbol_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            period = self.period_combo.currentText()
            
            # Get time range
            end_date = datetime.now()
            if period == "1 Month":
                start_date = end_date - timedelta(days=30)
            elif period == "3 Months":
                start_date = end_date - timedelta(days=90)
            elif period == "6 Months":
                start_date = end_date - timedelta(days=180)
            else:  # 1 Year
                start_date = end_date - timedelta(days=365)
                
            # Get market data
            data = self.trading_system.get_market_data(
                symbol, timeframe, start_time=start_date
            )
            
            if data.empty:
                print("No data available for market regime analysis")
                return
                
            # In a real implementation, you would perform actual market regime analysis
            # For demonstration, we'll simulate some analysis results
            
            # Define possible regimes
            regimes = ["Bull Trend", "Bear Trend", "Sideways", "High Volatility", "Low Volatility"]
            
            # Current regime (in a real system, this would be determined by analysis)
            current_regime = regimes[np.random.randint(0, len(regimes))]
            
            # Update current state labels
            self.regime_label.setText(current_regime)
            
            # Trend strength (0-1)
            trend_strength = np.random.random()
            self.trend_label.setText(f"{trend_strength:.2f}")
            
            # Volatility
            volatility_levels = ["Very Low", "Low", "Medium", "High", "Very High"]
            volatility = volatility_levels[np.random.randint(0, len(volatility_levels))]
            self.volatility_label.setText(volatility)
            
            # Momentum
            momentum_levels = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
            momentum = momentum_levels[np.random.randint(0, len(momentum_levels))]
            self.momentum_label.setText(momentum)
            
            # Price levels
            if 'close' in data.columns:
                current_price = data['close'].iloc[-1]
                
                # Simulate support and resistance
                support = current_price * 0.95
                resistance = current_price * 1.05
                
                self.support_label.setText(f"${support:.2f}")
                self.resistance_label.setText(f"${resistance:.2f}")
            
            # Update regime histogram
            self.regime_figure.clear()
            ax = self.regime_figure.add_subplot(111)
            
            # Generate regime distribution data
            regime_counts = {regime: np.random.randint(5, 60) for regime in regimes}
            
            # Create bar chart
            bars = ax.bar(regime_counts.keys(), regime_counts.values())
            
            # Color the bars
            colors = ['green', 'red', 'blue', 'orange', 'purple']
            for i, bar in enumerate(bars):
                bar.set_color(colors[i % len(colors)])
                
            ax.set_title("Market Regime Distribution")
            ax.set_ylabel("Days")
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            self.regime_figure.tight_layout()
            self.regime_canvas.draw()
            
            # Update transitions chart
            self.transitions_chart.clear()
            
            # Generate synthetic regime transition data
            days = (end_date - start_date).days
            regime_indices = []
            
            # Start with a random regime
            current_idx = np.random.randint(0, len(regimes))
            
            for _ in range(days):
                regime_indices.append(current_idx)
                
                # 10% chance of regime change each day
                if np.random.random() < 0.1:
                    # Choose a new regime
                    new_idx = np.random.randint(0, len(regimes))
                    while new_idx == current_idx:
                        new_idx = np.random.randint(0, len(regimes))
                        
                    current_idx = new_idx
            
            # Plot regime transitions
            pen = pg.mkPen(color='b', width=2)
            self.transitions_chart.plot(range(len(regime_indices)), regime_indices, pen=pen, symbol='o', symbolSize=5)
            
            # Set y-axis ticks to regime names
            y_axis = self.transitions_chart.getAxis('left')
            y_axis.setTicks([[(i, name) for i, name in enumerate(regimes)]])
            
            # Set x-axis label
            self.transitions_chart.setLabel('bottom', 'Days')
            
        except Exception as e:
            print(f"Error analyzing market regime: {e}")

class AnalyticsTab(QWidget):
    """Analytics tab showing various trading analytics and visualizations"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for different analytics
        self.analytics_tabs = QTabWidget()
        
        # Model Performance tab
        self.model_performance_widget = ModelPerformanceWidget(self.trading_system)
        self.analytics_tabs.addTab(self.model_performance_widget, "Model Performance")
        
        # Correlation Analysis tab
        self.correlation_widget = CorrelationMatrixWidget(self.trading_system)
        self.analytics_tabs.addTab(self.correlation_widget, "Correlation Analysis")
        
        # Portfolio Analysis tab
        self.portfolio_widget = PortfolioAnalysisWidget(self.trading_system)
        self.analytics_tabs.addTab(self.portfolio_widget, "Portfolio Analysis")
        
        # Market Regime Analysis tab
        self.market_regime_widget = MarketRegimeWidget(self.trading_system)
        self.analytics_tabs.addTab(self.market_regime_widget, "Market Regime")
        
        main_layout.addWidget(self.analytics_tabs)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh All")
        refresh_btn.clicked.connect(self.refresh_data)
        main_layout.addWidget(refresh_btn)
        
    def refresh_data(self):
        """Refresh all analytics data"""
        try:
            # Refresh the currently visible tab
            current_tab = self.analytics_tabs.currentWidget()
            
            if current_tab == self.model_performance_widget:
                self.model_performance_widget.update_model_performance()
            elif current_tab == self.correlation_widget:
                self.correlation_widget.update_correlation_matrix()
            elif current_tab == self.portfolio_widget:
                self.portfolio_widget.update_portfolio_analysis()
            elif current_tab == self.market_regime_widget:
                self.market_regime_widget.analyze_market_regime()
                
        except Exception as e:
            print(f"Error refreshing analytics data: {e}")