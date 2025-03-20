# dashboard_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QGridLayout, QPushButton, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from datetime import datetime, timedelta

class DashboardTab(QWidget):
    """Dashboard tab showing system overview and performance metrics"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Top section with portfolio overview
        top_layout = QHBoxLayout()
        
        # Portfolio value card
        self.portfolio_card = self._create_value_card("Portfolio Value", "$10,000.00")
        top_layout.addWidget(self.portfolio_card)
        
        # Daily P&L card
        self.daily_pnl_card = self._create_value_card("Daily P&L", "+$0.00", "+0.00%")
        top_layout.addWidget(self.daily_pnl_card)
        
        # Total P&L card
        self.total_pnl_card = self._create_value_card("Total P&L", "+$0.00", "+0.00%")
        top_layout.addWidget(self.total_pnl_card)
        
        # Open positions card
        self.positions_card = self._create_value_card("Open Positions", "0")
        top_layout.addWidget(self.positions_card)
        
        main_layout.addLayout(top_layout)
        
        # Middle section with charts
        middle_layout = QHBoxLayout()
        
        # Portfolio history chart
        portfolio_layout = QVBoxLayout()
        portfolio_layout.addWidget(QLabel("Portfolio History"))
        
        self.portfolio_chart = pg.PlotWidget()
        self.portfolio_chart.setBackground('w')
        self.portfolio_chart.showGrid(x=True, y=True)
        self.portfolio_chart.setLabel('left', 'Value ($)')
        self.portfolio_chart.setLabel('bottom', 'Time')
        
        portfolio_layout.addWidget(self.portfolio_chart)
        middle_layout.addLayout(portfolio_layout, 2)
        
        # Trading activity chart
        activity_layout = QVBoxLayout()
        activity_layout.addWidget(QLabel("Trading Activity"))
        
        self.activity_chart = pg.PlotWidget()
        self.activity_chart.setBackground('w')
        self.activity_chart.showGrid(x=True, y=True)
        
        activity_layout.addWidget(self.activity_chart)
        middle_layout.addLayout(activity_layout, 1)
        
        main_layout.addLayout(middle_layout)
        
        # Bottom section with trading stats and performance metrics
        bottom_layout = QHBoxLayout()
        
        # Trading stats
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QGridLayout(stats_frame)
        
        stats_layout.addWidget(QLabel("Win Rate:"), 0, 0)
        self.win_rate_label = QLabel("0.00%")
        stats_layout.addWidget(self.win_rate_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Profit Factor:"), 1, 0)
        self.profit_factor_label = QLabel("0.00")
        stats_layout.addWidget(self.profit_factor_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Avg. Win:"), 2, 0)
        self.avg_win_label = QLabel("$0.00")
        stats_layout.addWidget(self.avg_win_label, 2, 1)
        
        stats_layout.addWidget(QLabel("Avg. Loss:"), 3, 0)
        self.avg_loss_label = QLabel("$0.00")
        stats_layout.addWidget(self.avg_loss_label, 3, 1)
        
        bottom_layout.addWidget(stats_frame)
        
        # Performance metrics
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.StyledPanel)
        metrics_layout = QGridLayout(metrics_frame)
        
        metrics_layout.addWidget(QLabel("Sharpe Ratio:"), 0, 0)
        self.sharpe_label = QLabel("0.00")
        metrics_layout.addWidget(self.sharpe_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Max Drawdown:"), 1, 0)
        self.drawdown_label = QLabel("0.00%")
        metrics_layout.addWidget(self.drawdown_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Trades/Day:"), 2, 0)
        self.trades_per_day_label = QLabel("0.00")
        metrics_layout.addWidget(self.trades_per_day_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("ROI:"), 3, 0)
        self.roi_label = QLabel("0.00%")
        metrics_layout.addWidget(self.roi_label, 3, 1)
        
        bottom_layout.addWidget(metrics_frame)
        
        # Active strategies
        strategies_frame = QFrame()
        strategies_frame.setFrameShape(QFrame.StyledPanel)
        strategies_layout = QVBoxLayout(strategies_frame)
        
        strategies_layout.addWidget(QLabel("Active Strategies"))
        
        self.strategies_list = QLabel()
        self.strategies_list.setText("- None active")
        strategies_layout.addWidget(self.strategies_list)
        
        # Time frame selector
        time_frame_layout = QHBoxLayout()
        time_frame_layout.addWidget(QLabel("Time Frame:"))
        
        self.time_frame_combo = QComboBox()
        self.time_frame_combo.addItems(["1 Day", "1 Week", "1 Month", "3 Months", "All"])
        self.time_frame_combo.currentIndexChanged.connect(self.refresh_data)
        time_frame_layout.addWidget(self.time_frame_combo)
        
        strategies_layout.addLayout(time_frame_layout)
        
        bottom_layout.addWidget(strategies_frame)
        
        main_layout.addLayout(bottom_layout)
        
        # Initial data load
        self.refresh_data()
        
    def _create_value_card(self, title, value, subtitle=None):
        """Create a value display card"""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        
        layout = QVBoxLayout(card)
        
        # Title
        title_label = QLabel(title)
        layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(value)
        value_font = QFont()
        value_font.setPointSize(16)
        value_font.setBold(True)
        value_label.setFont(value_font)
        layout.addWidget(value_label)
        
        # Subtitle (optional)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            layout.addWidget(subtitle_label)
        
        # Store labels for updating
        card.value_label = value_label
        if subtitle:
            card.subtitle_label = subtitle_label
            
        return card
        
    def refresh_data(self):
        """Refresh all dashboard data"""
        try:
            # Get portfolio value
            portfolio = self.trading_system.get_portfolio_value()
            
            if portfolio:
                # Update portfolio card
                total_value = portfolio.get('total_value', 0)
                self.portfolio_card.value_label.setText(f"${total_value:.2f}")
                
                # Update PnL cards
                initial_capital = portfolio.get('initial_capital', 0)
                total_pnl = total_value - initial_capital
                total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
                
                self.total_pnl_card.value_label.setText(f"${total_pnl:.2f}")
                if hasattr(self.total_pnl_card, 'subtitle_label'):
                    sign = "+" if total_pnl >= 0 else ""
                    self.total_pnl_card.subtitle_label.setText(f"{sign}{total_pnl_pct:.2f}%")
                    
                # Update positions card
                open_positions = len(self.trading_system.open_positions)
                self.positions_card.value_label.setText(str(open_positions))
                
            # Update performance metrics
            self._update_performance_metrics()
            
            # Update charts
            self._update_charts()
            
            # Update strategies list
            active_strategies = self.trading_system.config["strategies"]["active_strategies"]
            if active_strategies:
                strategy_text = "\n".join([f"- {s}" for s in active_strategies])
                self.strategies_list.setText(strategy_text)
                
        except Exception as e:
            print(f"Error refreshing dashboard: {e}")
            
    def _update_performance_metrics(self):
        """Update performance metric displays"""
        # Get trade history
        trade_history = self.trading_system.trade_history
        
        if not trade_history:
            return
            
        # Calculate win rate
        wins = [trade for trade in trade_history if trade.get('pnl', 0) > 0]
        win_rate = len(wins) / len(trade_history) if trade_history else 0
        self.win_rate_label.setText(f"{win_rate * 100:.2f}%")
        
        # Calculate profit factor
        total_wins = sum(trade.get('pnl', 0) for trade in wins)
        losses = [trade for trade in trade_history if trade.get('pnl', 0) <= 0]
        total_losses = sum(abs(trade.get('pnl', 0)) for trade in losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        self.profit_factor_label.setText(f"{profit_factor:.2f}")
        
        # Calculate average win/loss
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        self.avg_win_label.setText(f"${avg_win:.2f}")
        self.avg_loss_label.setText(f"${avg_loss:.2f}")
        
        # TODO: Calculate Sharpe ratio
        self.sharpe_label.setText("0.00")
        
        # Calculate drawdown (simplified)
        self.drawdown_label.setText(f"{self.trading_system.position_sizer.max_drawdown * 100:.2f}%")
        
        # Calculate trades per day
        if len(trade_history) >= 2:
            first_trade_time = min(trade.get('entry_time', datetime.now()) for trade in trade_history)
            last_trade_time = max(trade.get('exit_time', datetime.now()) for trade in trade_history)
            days = (last_trade_time - first_trade_time).days + 1
            trades_per_day = len(trade_history) / days if days > 0 else 0
            self.trades_per_day_label.setText(f"{trades_per_day:.2f}")
            
        # Calculate ROI
        portfolio = self.trading_system.get_portfolio_value()
        if portfolio:
            total_value = portfolio.get('total_value', 0)
            initial_capital = portfolio.get('initial_capital', 0)
            roi = ((total_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0
            self.roi_label.setText(f"{roi:.2f}%")
            
    def _update_charts(self):
        """Update chart displays"""
        # Get time frame selection
        time_frame = self.time_frame_combo.currentText()
        
        # Calculate date range
        end_date = datetime.now()
        
        if time_frame == "1 Day":
            start_date = end_date - timedelta(days=1)
        elif time_frame == "1 Week":
            start_date = end_date - timedelta(weeks=1)
        elif time_frame == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif time_frame == "3 Months":
            start_date = end_date - timedelta(days=90)
        else:  # All
            start_date = datetime(2000, 1, 1)  # Far in the past
            
        # Filter trade history by date
        trade_history = [
            trade for trade in self.trading_system.trade_history
            if trade.get('entry_time', datetime.now()) >= start_date
        ]
        
        # Portfolio chart - simulated data for now
        self.portfolio_chart.clear()
        
        # TODO: Replace with actual portfolio history data
        dates = np.linspace(0, 100, 100)
        portfolio_values = 10000 + np.cumsum(np.random.normal(5, 20, 100))
        
        pen = pg.mkPen(color='b', width=2)
        self.portfolio_chart.plot(dates, portfolio_values, pen=pen)
        
        # Trading activity chart - plot trades
        self.activity_chart.clear()
        
        if trade_history:
            # Extract trade data
            entry_times = [i for i, trade in enumerate(trade_history)]
            pnls = [trade.get('pnl', 0) for trade in trade_history]
            
            # Create bar graph
            bar_item = pg.BarGraphItem(x=entry_times, height=pnls, width=0.6, brush='g')
            self.activity_chart.addItem(bar_item)
            
    def update_status(self):
        """Update status information"""
        # Update daily PnL (simulated for now)
        daily_pnl = 0
        daily_pnl_pct = 0
        
        # TODO: Calculate actual daily PnL
        
        self.daily_pnl_card.value_label.setText(f"${daily_pnl:.2f}")
        if hasattr(self.daily_pnl_card, 'subtitle_label'):
            sign = "+" if daily_pnl >= 0 else ""
            self.daily_pnl_card.subtitle_label.setText(f"{sign}{daily_pnl_pct:.2f}%")