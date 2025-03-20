# risk_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QTableWidget, QTableWidgetItem, QPushButton,
                           QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                           QHeaderView, QSplitter, QProgressBar, QSlider,
                           QSpinBox, QTabWidget, QTextEdit, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer

class RiskTab(QWidget):
    """Risk management tab for controlling and monitoring trading system risk"""
    
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
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # Top section - Risk Controls
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Position Sizing Controls
        position_group = QGroupBox("Position Sizing")
        position_layout = QFormLayout(position_group)
        
        # Base risk per trade
        self.base_risk_spin = QDoubleSpinBox()
        self.base_risk_spin.setRange(0.001, 0.1)
        self.base_risk_spin.setSingleStep(0.001)
        self.base_risk_spin.setDecimals(3)
        self.base_risk_spin.setValue(0.02)  # Default 2%
        if hasattr(self.trading_system, 'position_sizer') and hasattr(self.trading_system.position_sizer, 'base_risk'):
            self.base_risk_spin.setValue(self.trading_system.position_sizer.base_risk)
        self.base_risk_spin.valueChanged.connect(self._update_base_risk)
        position_layout.addRow("Base Risk Per Trade:", self.base_risk_spin)
        
        # Max risk per trade
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0.001, 0.2)
        self.max_risk_spin.setSingleStep(0.001)
        self.max_risk_spin.setDecimals(3)
        self.max_risk_spin.setValue(0.05)  # Default 5%
        if hasattr(self.trading_system, 'position_sizer') and hasattr(self.trading_system.position_sizer, 'max_risk'):
            self.max_risk_spin.setValue(self.trading_system.position_sizer.max_risk)
        self.max_risk_spin.valueChanged.connect(self._update_max_risk)
        position_layout.addRow("Max Risk Per Trade:", self.max_risk_spin)
        
        # Min risk per trade
        self.min_risk_spin = QDoubleSpinBox()
        self.min_risk_spin.setRange(0.001, 0.05)
        self.min_risk_spin.setSingleStep(0.001)
        self.min_risk_spin.setDecimals(3)
        self.min_risk_spin.setValue(0.01)  # Default 1%
        if hasattr(self.trading_system, 'position_sizer') and hasattr(self.trading_system.position_sizer, 'min_risk'):
            self.min_risk_spin.setValue(self.trading_system.position_sizer.min_risk)
        self.min_risk_spin.valueChanged.connect(self._update_min_risk)
        position_layout.addRow("Min Risk Per Trade:", self.min_risk_spin)
        
        # Kelly criterion toggle
        self.kelly_check = QCheckBox("Use Kelly Criterion")
        self.kelly_check.setChecked(False)
        self.kelly_check.stateChanged.connect(self._toggle_kelly)
        position_layout.addRow("", self.kelly_check)
        
        # Kelly fractional setting
        self.kelly_fraction_spin = QDoubleSpinBox()
        self.kelly_fraction_spin.setRange(0.1, 1.0)
        self.kelly_fraction_spin.setSingleStep(0.1)
        self.kelly_fraction_spin.setDecimals(1)
        self.kelly_fraction_spin.setValue(0.5)  # Default to half-Kelly
        self.kelly_fraction_spin.setEnabled(False)
        position_layout.addRow("Kelly Fraction:", self.kelly_fraction_spin)
        
        # Position Volatility Adjustment
        self.volatility_check = QCheckBox("Adjust for Volatility")
        self.volatility_check.setChecked(True)
        position_layout.addRow("", self.volatility_check)
        
        # Apply changes button
        self.apply_position_btn = QPushButton("Apply Position Sizing Changes")
        self.apply_position_btn.clicked.connect(self._apply_position_sizing)
        position_layout.addRow("", self.apply_position_btn)
        
        top_layout.addWidget(position_group)
        
        # Risk Limits Controls
        limits_group = QGroupBox("Risk Limits")
        limits_layout = QFormLayout(limits_group)
        
        # Maximum positions
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 20)
        self.max_positions_spin.setValue(5)  # Default 5 positions
        # Check if risk manager exists and has the attribute
        if hasattr(self.trading_system, 'risk_manager') and hasattr(self.trading_system.risk_manager, 'risk_limits'):
            # Set max positions from risk_limits if available
            max_positions = self.trading_system.risk_manager.risk_limits.get("position_limit", 5)
            self.max_positions_spin.setValue(max_positions)
        limits_layout.addRow("Max Open Positions:", self.max_positions_spin)
        
        # Max risk per sector/asset
        self.max_sector_risk_spin = QDoubleSpinBox()
        self.max_sector_risk_spin.setRange(0.01, 0.5)
        self.max_sector_risk_spin.setSingleStep(0.01)
        self.max_sector_risk_spin.setDecimals(2)
        self.max_sector_risk_spin.setValue(0.2)  # Default 20%
        limits_layout.addRow("Max Risk Per Asset:", self.max_sector_risk_spin)
        
        # Daily loss limit
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(0.01, 0.1)
        self.daily_loss_spin.setSingleStep(0.01)
        self.daily_loss_spin.setDecimals(2)
        self.daily_loss_spin.setValue(0.03)  # Default 3%
        self.daily_loss_spin.setSuffix("")
        limits_layout.addRow("Daily Loss Limit:", self.daily_loss_spin)
        
        # Max drawdown limit
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(0.05, 0.3)
        self.max_drawdown_spin.setSingleStep(0.01)
        self.max_drawdown_spin.setDecimals(2)
        self.max_drawdown_spin.setValue(0.15)  # Default 15%
        if hasattr(self.trading_system, 'risk_manager') and hasattr(self.trading_system.risk_manager, 'risk_limits'):
            # Set max drawdown from risk_limits if available
            max_drawdown = self.trading_system.risk_manager.risk_limits.get("max_drawdown_limit", 15) / 100.0
            self.max_drawdown_spin.setValue(max_drawdown)
        self.max_drawdown_spin.setSuffix("")
        limits_layout.addRow("Max Drawdown Limit:", self.max_drawdown_spin)
        
        # Correlation threshold
        self.correlation_spin = QDoubleSpinBox()
        self.correlation_spin.setRange(0.3, 0.9)
        self.correlation_spin.setSingleStep(0.05)
        self.correlation_spin.setDecimals(2)
        self.correlation_spin.setValue(0.7)  # Default 70%
        limits_layout.addRow("Correlation Threshold:", self.correlation_spin)
        
        # Apply changes button
        self.apply_limits_btn = QPushButton("Apply Risk Limits")
        self.apply_limits_btn.clicked.connect(self._apply_risk_limits)
        limits_layout.addRow("", self.apply_limits_btn)
        
        top_layout.addWidget(limits_group)
        
        # Risk Actions
        actions_group = QGroupBox("Risk Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Hedge button
        self.hedge_btn = QPushButton("Hedge Portfolio")
        self.hedge_btn.clicked.connect(self._hedge_portfolio)
        actions_layout.addWidget(self.hedge_btn)
        
        # Reduce exposure button
        self.reduce_btn = QPushButton("Reduce Exposure")
        self.reduce_btn.clicked.connect(self._reduce_exposure)
        actions_layout.addWidget(self.reduce_btn)
        
        # Close riskiest position button
        self.close_risky_btn = QPushButton("Close Riskiest Position")
        self.close_risky_btn.clicked.connect(self._close_riskiest)
        actions_layout.addWidget(self.close_risky_btn)
        
        # Force exit all button
        self.force_exit_btn = QPushButton("Force Exit All Positions")
        self.force_exit_btn.clicked.connect(self._force_exit_all)
        self.force_exit_btn.setStyleSheet("background-color: #ffcccc;")
        actions_layout.addWidget(self.force_exit_btn)
        
        top_layout.addWidget(actions_group)
        
        splitter.addWidget(top_widget)
        
        # Bottom section - Risk Statistics and Visualization Tabs
        bottom_tabs = QTabWidget()
        
        # Risk Metrics Tab
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)
        
        # Risk metrics tables
        self.metrics_table = QTableWidget(5, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        
        # Set up metrics rows
        metrics = [
            ("Portfolio Value at Risk (VaR)", "$0.00"),
            ("Current Drawdown", "0.00%"),
            ("Maximum Drawdown", "0.00%"),
            ("Risk-Reward Ratio", "0.00"),
            ("Portfolio Beta", "0.00")
        ]
        
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
            
        metrics_layout.addWidget(self.metrics_table)
        
        # Current Risk Factors
        factors_group = QGroupBox("Current Risk Factors")
        factors_layout = QGridLayout(factors_group)
        
        # Volatility factor
        factors_layout.addWidget(QLabel("Volatility:"), 0, 0)
        self.volatility_label = QLabel("1.00")
        factors_layout.addWidget(self.volatility_label, 0, 1)
        
        # Confidence factor
        factors_layout.addWidget(QLabel("Confidence:"), 0, 2)
        self.confidence_label = QLabel("1.00")
        factors_layout.addWidget(self.confidence_label, 0, 3)
        
        # Streak factor
        factors_layout.addWidget(QLabel("Streak:"), 1, 0)
        self.streak_label = QLabel("1.00")
        factors_layout.addWidget(self.streak_label, 1, 1)
        
        # Equity curve factor
        factors_layout.addWidget(QLabel("Equity Curve:"), 1, 2)
        self.equity_factor_label = QLabel("1.00")
        factors_layout.addWidget(self.equity_factor_label, 1, 3)
        
        # Market condition factor
        factors_layout.addWidget(QLabel("Market Condition:"), 2, 0)
        self.market_factor_label = QLabel("1.00")
        factors_layout.addWidget(self.market_factor_label, 2, 1)
        
        metrics_layout.addWidget(factors_group)
        
        # Add "Update Metrics" button
        refresh_btn = QPushButton("Refresh Risk Metrics")
        refresh_btn.clicked.connect(self.refresh_data)
        metrics_layout.addWidget(refresh_btn)
        
        bottom_tabs.addTab(self.metrics_tab, "Risk Metrics")
        
        # Portfolio Allocation Tab
        self.allocation_tab = QWidget()
        allocation_layout = QVBoxLayout(self.allocation_tab)
        
        # Current allocation table
        self.allocation_table = QTableWidget(0, 3)
        self.allocation_table.setHorizontalHeaderLabels(["Asset", "Allocation", "Risk Contribution"])
        self.allocation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.allocation_table.verticalHeader().setVisible(False)
        allocation_layout.addWidget(self.allocation_table)
        
        bottom_tabs.addTab(self.allocation_tab, "Portfolio Allocation")
        
        # Drawdown Analysis Tab
        self.drawdown_tab = QWidget()
        drawdown_layout = QVBoxLayout(self.drawdown_tab)
        
        # Drawdown statistics
        drawdown_stats_group = QGroupBox("Drawdown Statistics")
        drawdown_stats_layout = QFormLayout(drawdown_stats_group)
        
        self.max_dd_label = QLabel("0.00%")
        drawdown_stats_layout.addRow("Maximum Drawdown:", self.max_dd_label)
        
        self.avg_dd_label = QLabel("0.00%")
        drawdown_stats_layout.addRow("Average Drawdown:", self.avg_dd_label)
        
        self.dd_duration_label = QLabel("0 days")
        drawdown_stats_layout.addRow("Avg Drawdown Duration:", self.dd_duration_label)
        
        self.dd_recovery_label = QLabel("0 days")
        drawdown_stats_layout.addRow("Avg Recovery Time:", self.dd_recovery_label)
        
        drawdown_layout.addWidget(drawdown_stats_group)
        
        bottom_tabs.addTab(self.drawdown_tab, "Drawdown Analysis")
        
        # Risk Recommendations Tab
        self.recommendations_tab = QWidget()
        recommendations_layout = QVBoxLayout(self.recommendations_tab)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        recommendations_layout.addWidget(self.recommendations_text)
        
        # Apply recommendations button
        self.apply_recommendations_btn = QPushButton("Apply Recommendations")
        self.apply_recommendations_btn.clicked.connect(self._apply_recommendations)
        recommendations_layout.addWidget(self.apply_recommendations_btn)
        
        bottom_tabs.addTab(self.recommendations_tab, "Risk Recommendations")
        
        # Add tabs to splitter
        splitter.addWidget(bottom_tabs)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Initial data refresh
        self.refresh_data()
        
    def refresh_data(self):
        """Refresh all risk data"""
        try:
            # Update risk metrics
            self._update_risk_metrics()
            self._update_portfolio_allocation()
            self._update_risk_recommendations()
            self._update_risk_factors()
        except Exception as e:
            print(f"Error refreshing risk data: {e}")
            
    def _update_risk_metrics(self):
        """Update risk metrics display"""
        try:
            # Update VaR display (placeholder)
            self.metrics_table.setItem(0, 1, QTableWidgetItem("$150.25"))
            
            # Get drawdown data (placeholder)
            current_drawdown = 0.05  # 5% example
            max_drawdown = 0.12  # 12% example
            
            # Update drawdown displays
            self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{current_drawdown * 100:.2f}%"))
            self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{max_drawdown * 100:.2f}%"))
            
            # Update other metrics
            self.metrics_table.setItem(3, 1, QTableWidgetItem("2.15"))
            self.metrics_table.setItem(4, 1, QTableWidgetItem("1.20"))
            
            # Update drawdown statistics tab
            self.max_dd_label.setText(f"{max_drawdown * 100:.2f}%")
            self.avg_dd_label.setText("6.25%")
            self.dd_duration_label.setText("7 days")
            self.dd_recovery_label.setText("12 days")
        except Exception as e:
            print(f"Error updating risk metrics: {e}")
            
    def _update_portfolio_allocation(self):
        """Update portfolio allocation display"""
        try:
            # Clear allocation table
            self.allocation_table.setRowCount(0)
            
            # Example allocation data
            allocations = [
                ("BTC", 0.35, 0.42),
                ("ETH", 0.25, 0.28),
                ("SOL", 0.15, 0.18),
                ("CASH", 0.25, 0.12)
            ]
            
            # Fill allocation table
            for i, (asset, allocation, risk) in enumerate(allocations):
                self.allocation_table.insertRow(i)
                self.allocation_table.setItem(i, 0, QTableWidgetItem(asset))
                self.allocation_table.setItem(i, 1, QTableWidgetItem(f"{allocation * 100:.2f}%"))
                self.allocation_table.setItem(i, 2, QTableWidgetItem(f"{risk * 100:.2f}%"))
        except Exception as e:
            print(f"Error updating portfolio allocation: {e}")
            
    def _update_risk_recommendations(self):
        """Update risk recommendations"""
        try:
            # Example recommendations
            text = "Risk Management Recommendations:\n\n"
            text += "⚠ MEDIUM PRIORITY: Consider reducing position sizes due to increased market volatility\n\n"
            text += "⚠ MEDIUM PRIORITY: High correlation detected between BTC and ETH positions (0.85)\n\n"
            text += "• Daily loss limit approaching threshold, monitor closely"
            
            self.recommendations_text.setText(text)
        except Exception as e:
            print(f"Error updating risk recommendations: {e}")
            
    def _update_risk_factors(self):
        """Update risk factor displays"""
        try:
            # Example risk factors
            self.volatility_label.setText("1.25")
            self.confidence_label.setText("0.85")
            self.streak_label.setText("0.95")
            self.equity_factor_label.setText("1.10")
            self.market_factor_label.setText("0.90")
        except Exception as e:
            print(f"Error updating risk factors: {e}")
            
    def _update_base_risk(self, value):
        """Update base risk per trade"""
        try:
            # Ensure max_risk > base_risk > min_risk
            if value > self.max_risk_spin.value():
                self.max_risk_spin.setValue(value)
            if value < self.min_risk_spin.value():
                self.min_risk_spin.setValue(value)
        except Exception as e:
            print(f"Error updating base risk: {e}")
            
    def _update_max_risk(self, value):
        """Update max risk per trade"""
        try:
            # Ensure max_risk > base_risk
            if value < self.base_risk_spin.value():
                self.base_risk_spin.setValue(value)
        except Exception as e:
            print(f"Error updating max risk: {e}")
            
    def _update_min_risk(self, value):
        """Update min risk per trade"""
        try:
            # Ensure min_risk < base_risk
            if value > self.base_risk_spin.value():
                self.base_risk_spin.setValue(value)
        except Exception as e:
            print(f"Error updating min risk: {e}")
            
    def _toggle_kelly(self, state):
        """Toggle Kelly criterion usage"""
        self.kelly_fraction_spin.setEnabled(state == Qt.Checked)
        
    def _apply_position_sizing(self):
        """Apply position sizing changes"""
        try:
            if hasattr(self.trading_system, 'position_sizer'):
                # Update position sizer if it exists
                if hasattr(self.trading_system.position_sizer, 'base_risk'):
                    self.trading_system.position_sizer.base_risk = self.base_risk_spin.value()
                if hasattr(self.trading_system.position_sizer, 'max_risk'):
                    self.trading_system.position_sizer.max_risk = self.max_risk_spin.value()
                if hasattr(self.trading_system.position_sizer, 'min_risk'):
                    self.trading_system.position_sizer.min_risk = self.min_risk_spin.value()
                
            print("Applied position sizing changes")
        except Exception as e:
            print(f"Error applying position sizing changes: {e}")
            
    def _apply_risk_limits(self):
        """Apply risk limits changes"""
        try:
            if hasattr(self.trading_system, 'risk_manager'):
                # Update risk limits
                if hasattr(self.trading_system.risk_manager, 'risk_limits'):
                    limits = self.trading_system.risk_manager.risk_limits
                    limits["position_limit"] = self.max_positions_spin.value()
                    limits["max_drawdown_limit"] = self.max_drawdown_spin.value() * 100  # Convert to percentage
                
                if hasattr(self.trading_system.risk_manager, 'set_risk_limits'):
                    # Use dedicated setter if available
                    self.trading_system.risk_manager.set_risk_limits(limits)
                    
            print("Applied risk limits changes")
        except Exception as e:
            print(f"Error applying risk limits changes: {e}")
            
    def _hedge_portfolio(self):
        """Implement hedging strategy"""
        print("Hedging portfolio...")
            
    def _reduce_exposure(self):
        """Reduce overall portfolio exposure"""
        print("Reducing exposure...")
            
    def _close_riskiest(self):
        """Close the riskiest position"""
        print("Closing riskiest position...")
            
    def _force_exit_all(self):
        """Force exit all positions"""
        print("Forcing exit of all positions...")
            
    def _apply_recommendations(self):
        """Apply risk recommendations"""
        print("Applying risk recommendations...")
