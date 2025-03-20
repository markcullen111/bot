# gui/widgets/strategy_selector_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QCheckBox, QDoubleSpinBox, QPushButton,
                           QGroupBox, QFormLayout, QSlider, QTableWidget,
                           QTableWidgetItem, QHeaderView, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal

class StrategyParameterWidget(QWidget):
    """Widget for editing a single strategy parameter"""
    
    parameter_changed = pyqtSignal(str, object)
    
    def __init__(self, param_name, param_value, param_type="float", parent=None):
        super().__init__(parent)
        
        self.param_name = param_name
        self.param_type = param_type
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Parameter label
        label = QLabel(param_name.replace('_', ' ').title() + ":")
        layout.addWidget(label)
        
        # Create appropriate editor based on parameter type
        if param_type == "float":
            self.editor = QDoubleSpinBox()
            self.editor.setDecimals(3)
            self.editor.setRange(0, 10)
            self.editor.setSingleStep(0.01)
            self.editor.setValue(float(param_value))
            self.editor.valueChanged.connect(self._on_value_changed)
        elif param_type == "int":
            self.editor = QDoubleSpinBox()
            self.editor.setDecimals(0)
            self.editor.setRange(0, 1000)
            self.editor.setValue(int(param_value))
            self.editor.valueChanged.connect(self._on_value_changed)
        elif param_type == "bool":
            self.editor = QCheckBox()
            self.editor.setChecked(bool(param_value))
            self.editor.stateChanged.connect(self._on_value_changed)
        else:
            # Default to string/text representation
            self.editor = QLineEdit(str(param_value))
            self.editor.textChanged.connect(self._on_value_changed)
            
        layout.addWidget(self.editor)
        
    def _on_value_changed(self, value):
        """Handle value changes"""
        self.parameter_changed.emit(self.param_name, value)
        
    def get_value(self):
        """Get current parameter value"""
        if self.param_type == "float":
            return self.editor.value()
        elif self.param_type == "int":
            return int(self.editor.value())
        elif self.param_type == "bool":
            return self.editor.isChecked()
        else:
            return self.editor.text()

class StrategySelector(QWidget):
    """
    Widget for selecting and configuring trading strategies.
    Allows users to enable/disable strategies and adjust their parameters.
    """
    
    strategy_changed = pyqtSignal(str, bool)  # Strategy name, enabled state
    parameters_changed = pyqtSignal(str, dict)  # Strategy name, parameters dict
    weight_changed = pyqtSignal(str, float)  # Strategy name, weight
    optimize_requested = pyqtSignal(str)  # Strategy name
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.strategies = {}  # Strategy data
        self.param_widgets = {}  # Parameter widgets
        
        # Initialize UI
        self._init_ui()
        
        # Load available strategies
        self._load_strategies()
        
    def _init_ui(self):
        """Initialize the UI components"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create strategy selection area
        selection_group = QGroupBox("Available Strategies")
        selection_layout = QVBoxLayout(selection_group)
        
        # Strategy selector and controls
        self.strategy_table = QTableWidget(0, 3)
        self.strategy_table.setHorizontalHeaderLabels(["Strategy", "Enabled", "Weight"])
        self.strategy_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.strategy_table.verticalHeader().setVisible(False)
        self.strategy_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.strategy_table.setSelectionMode(QTableWidget.SingleSelection)
        self.strategy_table.selectionModel().selectionChanged.connect(self._on_strategy_selected)
        
        selection_layout.addWidget(self.strategy_table)
        
        # Strategy action buttons
        actions_layout = QHBoxLayout()
        
        self.optimize_btn = QPushButton("Optimize Strategy")
        self.optimize_btn.clicked.connect(self._on_optimize_clicked)
        actions_layout.addWidget(self.optimize_btn)
        
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        actions_layout.addWidget(self.reset_btn)
        
        selection_layout.addLayout(actions_layout)
        
        layout.addWidget(selection_group)
        
        # Parameters section
        params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QFormLayout(params_group)
        
        # Add placeholder message
        self.placeholder_label = QLabel("Select a strategy to view parameters")
        self.params_layout.addRow(self.placeholder_label)
        
        layout.addWidget(params_group)
        
        # Initially disable parameter section
        params_group.setEnabled(False)
        self.params_group = params_group
        
    def _load_strategies(self):
        """Load available strategies from the trading system"""
        if not hasattr(self.trading_system, 'config'):
            return
            
        # Get active strategies from config
        active_strategies = self.trading_system.config.get("strategies", {}).get("active_strategies", [])
        
        # Clear the table
        self.strategy_table.setRowCount(0)
        
        # Get strategy parameters from RL manager if available
        strategy_params = {}
        if hasattr(self.trading_system, 'rl_manager') and hasattr(self.trading_system.rl_manager, 'strategy_params'):
            strategy_params = self.trading_system.rl_manager.strategy_params
            
        # Get strategy weights - default to equal weights
        if active_strategies:
            default_weight = 1.0 / len(active_strategies)
        else:
            default_weight = 1.0
            
        # Add strategies to table
        for i, strategy_name in enumerate(self.trading_system.strategy_system.strategy_generator.strategies.keys()):
            # Add row
            self.strategy_table.insertRow(i)
            
            # Strategy name
            self.strategy_table.setItem(i, 0, QTableWidgetItem(strategy_name))
            
            # Enabled checkbox
            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(strategy_name in active_strategies)
            enabled_checkbox.stateChanged.connect(
                lambda state, name=strategy_name: self._on_strategy_enabled(name, state)
            )
            self.strategy_table.setCellWidget(i, 1, self._wrap_widget(enabled_checkbox))
            
            # Weight spinner
            weight_spinner = QDoubleSpinBox()
            weight_spinner.setRange(0, 1)
            weight_spinner.setDecimals(2)
            weight_spinner.setSingleStep(0.05)
            weight_spinner.setValue(strategy_params.get(strategy_name, {}).get('weight', default_weight))
            weight_spinner.valueChanged.connect(
                lambda value, name=strategy_name: self._on_weight_changed(name, value)
            )
            self.strategy_table.setCellWidget(i, 2, self._wrap_widget(weight_spinner))
            
            # Store strategy data
            self.strategies[strategy_name] = {
                'enabled': strategy_name in active_strategies,
                'weight': strategy_params.get(strategy_name, {}).get('weight', default_weight),
                'parameters': strategy_params.get(strategy_name, {})
            }
            
    def _wrap_widget(self, widget):
        """Wrap a widget in a container for table cell placement"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignCenter)
        return container
        
    def _on_strategy_enabled(self, strategy_name, state):
        """Handle strategy enabled/disabled"""
        enabled = state == Qt.Checked
        self.strategies[strategy_name]['enabled'] = enabled
        self.strategy_changed.emit(strategy_name, enabled)
        
    def _on_weight_changed(self, strategy_name, weight):
        """Handle strategy weight change"""
        self.strategies[strategy_name]['weight'] = weight
        self.weight_changed.emit(strategy_name, weight)
        
    def _on_strategy_selected(self):
        """Handle strategy selection change"""
        selected_items = self.strategy_table.selectedItems()
        if not selected_items:
            # Clear parameters
            self._clear_parameters()
            self.params_group.setEnabled(False)
            return
            
        # Get selected strategy
        row = selected_items[0].row()
        strategy_name = self.strategy_table.item(row, 0).text()
        
        # Update parameters section
        self._update_parameters(strategy_name)
        self.params_group.setEnabled(True)
        
    def _update_parameters(self, strategy_name):
        """Update parameters section for selected strategy"""
        # Clear current parameters
        self._clear_parameters()
        
        # Get strategy parameters
        if strategy_name not in self.strategies:
            return
            
        # Get default parameters
        default_params = self._get_default_parameters(strategy_name)
        
        # Get current parameters
        current_params = self.strategies[strategy_name]['parameters']
        
        # Merge default with current
        params = {**default_params, **current_params}
        
        # Remove non-parameter keys
        exclude_keys = ['weight']
        params = {k: v for k, v in params.items() if k not in exclude_keys}
        
        # Create parameter widgets
        self.param_widgets = {}
        for param_name, param_value in params.items():
            # Determine parameter type
            param_type = "float"
            if isinstance(param_value, int):
                param_type = "int"
            elif isinstance(param_value, bool):
                param_type = "bool"
                
            # Create widget
            param_widget = StrategyParameterWidget(param_name, param_value, param_type)
            param_widget.parameter_changed.connect(
                lambda name, value, strat=strategy_name: self._on_parameter_changed(strat, name, value)
            )
            
            # Add to layout
            self.params_layout.addRow(param_widget)
            
            # Store widget
            self.param_widgets[param_name] = param_widget
            
    def _clear_parameters(self):
        """Clear parameters section"""
        # Remove all rows except the first
        while self.params_layout.rowCount() > 0:
            # Get widget from row
            item = self.params_layout.itemAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                self.params_layout.removeItem(item)
            else:
                break
                
        # Add placeholder
        self.placeholder_label = QLabel("Select a strategy to view parameters")
        self.params_layout.addRow(self.placeholder_label)
        
    def _on_parameter_changed(self, strategy_name, param_name, value):
        """Handle parameter value change"""
        # Update strategy parameters
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['parameters'][param_name] = value
            
            # Emit signal
            self.parameters_changed.emit(strategy_name, self.strategies[strategy_name]['parameters'])
            
    def _on_optimize_clicked(self):
        """Handle optimize button click"""
        selected_items = self.strategy_table.selectedItems()
        if not selected_items:
            return
            
        # Get selected strategy
        row = selected_items[0].row()
        strategy_name = self.strategy_table.item(row, 0).text()
        
        # Emit optimize request
        self.optimize_requested.emit(strategy_name)
        
    def _on_reset_clicked(self):
        """Handle reset parameters button click"""
        selected_items = self.strategy_table.selectedItems()
        if not selected_items:
            return
            
        # Get selected strategy
        row = selected_items[0].row()
        strategy_name = self.strategy_table.item(row, 0).text()
        
        # Reset to default parameters
        default_params = self._get_default_parameters(strategy_name)
        
        # Update strategy
        self.strategies[strategy_name]['parameters'] = default_params.copy()
        
        # Update UI
        self._update_parameters(strategy_name)
        
        # Emit parameters changed
        self.parameters_changed.emit(strategy_name, default_params)
        
    def _get_default_parameters(self, strategy_name):
        """Get default parameters for a strategy"""
        # This would typically come from the strategy system
        # For now, use hard-coded defaults based on strategy type
        
        if strategy_name == 'trend_following':
            return {
                'short_window': 10,
                'medium_window': 50,
                'long_window': 100,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
        elif strategy_name == 'mean_reversion':
            return {
                'window': 20,
                'std_dev': 2.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        elif strategy_name == 'breakout':
            return {
                'window': 50,
                'volume_window': 10,
                'volume_threshold': 1.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        else:
            # Generic default parameters
            return {
                'window': 20,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
            
    def update_strategies(self):
        """Update strategy list and data"""
        self._load_strategies()
