# risk_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox)
from PyQt5.QtCore import Qt, QTimer

class RiskTab(QWidget):
    """Risk management tab for controlling and monitoring trading system risk"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self._init_ui()
        
        # Set up refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Position Sizing Controls
        position_group = QGroupBox("Position Sizing")
        position_layout = QFormLayout(position_group)
        
        # Base risk per trade
        self.base_risk_spin = QDoubleSpinBox()
        self.base_risk_spin.setRange(0.001, 0.1)
        self.base_risk_spin.setSingleStep(0.001)
        self.base_risk_spin.setDecimals(3)
        self.base_risk_spin.setValue(self.trading_system.position_sizer.base_risk)
        self.base_risk_spin.valueChanged.connect(self._update_base_risk)
        position_layout.addRow("Base Risk Per Trade:", self.base_risk_spin)
        
        # Kelly criterion toggle
        self.kelly_check = QCheckBox("Use Kelly Criterion")
        self.kelly_check.setChecked(False)
        self.kelly_check.stateChanged.connect(self._toggle_kelly)
        position_layout.addRow("", self.kelly_check)
        
        # Apply changes button
        self.apply_position_btn = QPushButton("Apply Position Sizing Changes")
        self.apply_position_btn.clicked.connect(self._apply_position_sizing)
        position_layout.addRow("", self.apply_position_btn)
        
        main_layout.addWidget(position_group)
        
        # Risk Limits Controls
        risk_group = QGroupBox("Risk Limits")
        risk_layout = QFormLayout(risk_group)
        
        # Max risk per sector (Fixing incorrect attribute reference)
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0.001, 0.2)
        self.max_risk_spin.setSingleStep(0.001)
        self.max_risk_spin.setDecimals(3)
        self.max_risk_spin.setValue(self.trading_system.risk_manager.max_risk_per_sector)
        risk_layout.addRow("Max Risk Per Sector:", self.max_risk_spin)
        
        # Max drawdown limit (Correct attribute reference)
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(0.05, 0.3)
        self.max_drawdown_spin.setSingleStep(0.01)
        self.max_drawdown_spin.setDecimals(2)
        self.max_drawdown_spin.setValue(self.trading_system.risk_manager.risk_limits['max_drawdown_limit'])
        risk_layout.addRow("Max Drawdown Limit:", self.max_drawdown_spin)
        
        # Apply changes button
        self.apply_risk_btn = QPushButton("Apply Risk Limits")
        self.apply_risk_btn.clicked.connect(self._apply_risk_limits)
        risk_layout.addRow("", self.apply_risk_btn)
        
        main_layout.addWidget(risk_group)
        
        self.setLayout(main_layout)
        
    def _update_base_risk(self, value):
        """Update base risk per trade"""
        self.trading_system.position_sizer.base_risk = value
    
    def _toggle_kelly(self, state):
        """Toggle Kelly criterion usage"""
        pass  # Add Kelly Criterion logic here
    
    def _apply_position_sizing(self):
        """Apply position sizing changes"""
        print("Applied position sizing changes")
        
    def _apply_risk_limits(self):
        """Apply risk limits changes"""
        self.trading_system.risk_manager.max_risk_per_sector = self.max_risk_spin.value()
        self.trading_system.risk_manager.max_drawdown_limit = self.max_drawdown_spin.value()
        print("Applied risk limits changes")
    
    def refresh_data(self):
        """Refresh risk data"""
        pass  # Add logic to refresh data when needed

