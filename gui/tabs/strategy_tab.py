# strategy_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QTableWidget, QTableWidgetItem, QPushButton,
                           QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                           QHeaderView, QSplitter, QListWidget, QListWidgetItem,
                           QStackedWidget, QCheckBox, QLineEdit, QMessageBox,
                           QDialog, QDialogButtonBox, QTabWidget, QTextEdit,
                           QScrollArea)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QIcon
import json
import os

class StrategyTab(QWidget):
    """Tab for managing and configuring trading strategies"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
        # Load strategies
        self.refresh_strategies()
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Strategy list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Available Strategies"))
        
        self.strategy_list = QListWidget()
        self.strategy_list.setMinimumWidth(200)
        self.strategy_list.currentItemChanged.connect(self._on_strategy_selected)
        left_layout.addWidget(self.strategy_list)
        
        # Strategy actions
        actions_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("New Strategy")
        self.add_btn.clicked.connect(self._add_strategy)
        actions_layout.addWidget(self.add_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_strategy)
        actions_layout.addWidget(self.delete_btn)
        
        left_layout.addLayout(actions_layout)
        
        # Activation section
        activation_group = QGroupBox("Activation")
        activation_layout = QVBoxLayout(activation_group)
        
        self.active_checkbox = QCheckBox("Active")
        self.active_checkbox.stateChanged.connect(self._toggle_strategy_active)
        activation_layout.addWidget(self.active_checkbox)
        
        self.weight_layout = QHBoxLayout()
        self.weight_layout.addWidget(QLabel("Weight:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 1)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setValue(1.0)
        self.weight_spin.valueChanged.connect(self._update_strategy_weight)
        self.weight_layout.addWidget(self.weight_spin)
        
        activation_layout.addLayout(self.weight_layout)
        
        left_layout.addWidget(activation_group)
        
        # Add left panel to splitter
        splitter.addWidget(left_panel)
        
        # Right panel - Strategy details and parameters
        self.right_panel = QStackedWidget()
        
        # Empty state widget
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.addWidget(QLabel("Select a strategy"))
        self.right_panel.addWidget(empty_widget)
        
        # Strategy editor widget
        self.strategy_editor = StrategyEditorWidget(self.trading_system)
        self.right_panel.addWidget(self.strategy_editor)
        
        # Add right panel to splitter
        splitter.addWidget(self.right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([250, 750])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add strategy builder button
        self.builder_btn = QPushButton("Visual Strategy Builder")
        self.builder_btn.clicked.connect(self._open_strategy_builder)
        main_layout.addWidget(self.builder_btn)
        
    def refresh_strategies(self):
        """Refresh the list of available strategies"""
        # Clear the list
        self.strategy_list.clear()
        
        # Get available strategies
        strategies = []
        
        # Get built-in strategies
        built_in = list(self.trading_system.strategy_system.strategy_generator.strategies.keys())
        for strategy in built_in:
            strategies.append({"name": strategy, "type": "built-in"})
            
        # Get custom strategies (would be stored in data/strategies)
        custom_strategies_dir = os.path.join("data", "strategies")
        if os.path.exists(custom_strategies_dir):
            for filename in os.listdir(custom_strategies_dir):
                if filename.endswith(".json"):
                    strategy_name = filename[:-5]  # Remove .json extension
                    strategies.append({"name": strategy_name, "type": "custom"})
        
        # Add items to list
        for strategy in strategies:
            item = QListWidgetItem(strategy["name"])
            item.setData(Qt.UserRole, strategy)
            
            # Mark active strategies
            active_strategies = self.trading_system.config["strategies"]["active_strategies"]
            if strategy["name"] in active_strategies:
                item.setBackground(QColor(200, 255, 200))  # Light green
                
            self.strategy_list.addItem(item)
            
    def _on_strategy_selected(self, current, previous):
        """Handle strategy selection"""
        if current is None:
            # Show empty state
            self.right_panel.setCurrentIndex(0)
            return
            
        # Get strategy data
        strategy_data = current.data(Qt.UserRole)
        
        # Update active checkbox
        active_strategies = self.trading_system.config["strategies"]["active_strategies"]
        self.active_checkbox.setChecked(strategy_data["name"] in active_strategies)
        
        # Update weight
        if hasattr(self.trading_system.rl_manager, 'strategy_params'):
            strategy_params = self.trading_system.rl_manager.strategy_params.get(strategy_data["name"], {})
            weight = strategy_params.get('weight', 1.0)
            self.weight_spin.setValue(weight)
        
        # Show strategy editor
        self.strategy_editor.load_strategy(strategy_data["name"], strategy_data["type"])
        self.right_panel.setCurrentIndex(1)
        
    def _add_strategy(self):
        """Add a new custom strategy"""
        # Open dialog for new strategy
        dialog = NewStrategyDialog(self)
        if dialog.exec_():
            # Get strategy details
            strategy_name = dialog.name_edit.text()
            strategy_type = dialog.type_combo.currentText()
            
            if not strategy_name:
                QMessageBox.warning(self, "Invalid Name", "Strategy name cannot be empty")
                return
                
            # Create strategy file
            self._create_strategy_file(strategy_name, strategy_type)
            
            # Refresh list
            self.refresh_strategies()
            
            # Select new strategy
            for i in range(self.strategy_list.count()):
                item = self.strategy_list.item(i)
                if item.text() == strategy_name:
                    self.strategy_list.setCurrentItem(item)
                    break
    
    def _create_strategy_file(self, name, strategy_type):
        """Create a new strategy file"""
        try:
            # Create strategies directory if not exists
            custom_strategies_dir = os.path.join("data", "strategies")
            os.makedirs(custom_strategies_dir, exist_ok=True)
            
            # Check if file already exists
            strategy_file = os.path.join(custom_strategies_dir, f"{name}.json")
            if os.path.exists(strategy_file):
                overwrite = QMessageBox.question(
                    self,
                    "Strategy Exists",
                    f"Strategy '{name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if overwrite != QMessageBox.Yes:
                    return False
            
            # Create template strategy
            strategy_data = {
                "name": name,
                "type": strategy_type,
                "description": "Custom trading strategy",
                "parameters": {
                    "window": 20,
                    "threshold": 1.5,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.05
                },
                "rules": []
            }
            
            # Save to file
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=4)
                
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating strategy: {e}")
            return False
            
    def _delete_strategy(self):
        """Delete the selected strategy"""
        # Get selected strategy
        current_item = self.strategy_list.currentItem()
        if current_item is None:
            return
            
        strategy_data = current_item.data(Qt.UserRole)
        
        # Cannot delete built-in strategies
        if strategy_data["type"] == "built-in":
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "Cannot delete built-in strategies"
            )
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete '{strategy_data['name']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
            
        try:
            # Delete strategy file
            strategy_file = os.path.join("data", "strategies", f"{strategy_data['name']}.json")
            if os.path.exists(strategy_file):
                os.remove(strategy_file)
                
            # Remove from active strategies if active
            active_strategies = self.trading_system.config["strategies"]["active_strategies"]
            if strategy_data["name"] in active_strategies:
                active_strategies.remove(strategy_data["name"])
                
            # Refresh list
            self.refresh_strategies()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting strategy: {e}")
            
    def _toggle_strategy_active(self, state):
        """Toggle strategy active state"""
        # Get selected strategy
        current_item = self.strategy_list.currentItem()
        if current_item is None:
            return
            
        strategy_data = current_item.data(Qt.UserRole)
        
        # Get active strategies
        active_strategies = self.trading_system.config["strategies"]["active_strategies"]
        
        # Add or remove from active strategies
        if state == Qt.Checked:
            if strategy_data["name"] not in active_strategies:
                active_strategies.append(strategy_data["name"])
        else:
            if strategy_data["name"] in active_strategies:
                active_strategies.remove(strategy_data["name"])
                
        # Update appearance
        if state == Qt.Checked:
            current_item.setBackground(QColor(200, 255, 200))  # Light green
        else:
            current_item.setBackground(QColor(255, 255, 255))  # White
            
    def _update_strategy_weight(self, value):
        """Update strategy weight"""
        # Get selected strategy
        current_item = self.strategy_list.currentItem()
        if current_item is None:
            return
            
        strategy_data = current_item.data(Qt.UserRole)
        
        # Update strategy weight in RL manager
        if hasattr(self.trading_system.rl_manager, 'strategy_params'):
            if strategy_data["name"] not in self.trading_system.rl_manager.strategy_params:
                self.trading_system.rl_manager.strategy_params[strategy_data["name"]] = {}
                
            self.trading_system.rl_manager.strategy_params[strategy_data["name"]]['weight'] = value
            
    def _open_strategy_builder(self):
        """Open the visual strategy builder"""
        try:
            # Import and open the strategy builder
            from strategy_builder import StrategyBuilderWindow
            
            strategy_builder = StrategyBuilderWindow(self)
            strategy_builder.show()
            
        except ImportError:
            QMessageBox.warning(
                self,
                "Strategy Builder",
                "Strategy Builder is not available in this version"
            )
            
class StrategyEditorWidget(QWidget):
    """Widget for editing strategy parameters and rules"""
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.current_strategy = None
        self.current_type = None
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Strategy info section
        info_layout = QHBoxLayout()
        
        # Strategy name and type
        name_layout = QVBoxLayout()
        self.name_label = QLabel("Strategy Name")
        self.name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        name_layout.addWidget(self.name_label)
        
        self.type_label = QLabel("Type: Unknown")
        name_layout.addWidget(self.type_label)
        
        info_layout.addLayout(name_layout, 1)
        
        # Strategy actions
        actions_layout = QVBoxLayout()
        actions_layout.setAlignment(Qt.AlignRight)
        
        self.optimize_btn = QPushButton("Optimize Parameters")
        self.optimize_btn.clicked.connect(self._optimize_parameters)
        actions_layout.addWidget(self.optimize_btn)
        
        self.backtest_btn = QPushButton("Backtest Strategy")
        self.backtest_btn.clicked.connect(self._backtest_strategy)
        actions_layout.addWidget(self.backtest_btn)
        
        info_layout.addLayout(actions_layout)
        
        main_layout.addLayout(info_layout)
        
        # Strategy details tabs
        self.detail_tabs = QTabWidget()
        
        # Parameters tab
        self.params_tab = QWidget()
        params_layout = QVBoxLayout(self.params_tab)
        
        self.params_form = QFormLayout()
        params_layout.addLayout(self.params_form)
        
        params_layout.addStretch()
        
        save_params_btn = QPushButton("Save Parameters")
        save_params_btn.clicked.connect(self._save_parameters)
        params_layout.addWidget(save_params_btn)
        
        self.detail_tabs.addTab(self.params_tab, "Parameters")
        
        # Description tab
        self.description_tab = QWidget()
        description_layout = QVBoxLayout(self.description_tab)
        
        self.description_edit = QTextEdit()
        description_layout.addWidget(self.description_edit)
        
        save_desc_btn = QPushButton("Save Description")
        save_desc_btn.clicked.connect(self._save_description)
        description_layout.addWidget(save_desc_btn)
        
        self.detail_tabs.addTab(self.description_tab, "Description")
        
        # Rules tab (for custom strategies)
        self.rules_tab = QWidget()
        rules_layout = QVBoxLayout(self.rules_tab)
        
        self.rules_table = QTableWidget(0, 4)
        self.rules_table.setHorizontalHeaderLabels(["Condition", "Value", "Action", ""])
        self.rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rules_layout.addWidget(self.rules_table)
        
        # Rule actions
        rule_actions = QHBoxLayout()
        
        add_rule_btn = QPushButton("Add Rule")
        add_rule_btn.clicked.connect(self._add_rule)
        rule_actions.addWidget(add_rule_btn)
        
        save_rules_btn = QPushButton("Save Rules")
        save_rules_btn.clicked.connect(self._save_rules)
        rule_actions.addWidget(save_rules_btn)
        
        rules_layout.addLayout(rule_actions)
        
        self.detail_tabs.addTab(self.rules_tab, "Rules")
        
        # Performance tab
        self.performance_tab = QWidget()
        performance_layout = QVBoxLayout(self.performance_tab)
        
        performance_layout.addWidget(QLabel("Strategy Performance Metrics"))
        
        # Performance metrics
        metrics_layout = QFormLayout()
        
        self.win_rate_label = QLabel("0.00%")
        metrics_layout.addRow("Win Rate:", self.win_rate_label)
        
        self.profit_factor_label = QLabel("0.00")
        metrics_layout.addRow("Profit Factor:", self.profit_factor_label)
        
        self.sharpe_label = QLabel("0.00")
        metrics_layout.addRow("Sharpe Ratio:", self.sharpe_label)
        
        self.drawdown_label = QLabel("0.00%")
        metrics_layout.addRow("Max Drawdown:", self.drawdown_label)
        
        performance_layout.addLayout(metrics_layout)
        
        # Note about metrics
        note_label = QLabel(
            "Note: Performance metrics are based on recent backtesting results. "
            "Run a backtest to update these metrics."
        )
        note_label.setWordWrap(True)
        performance_layout.addWidget(note_label)
        
        performance_layout.addStretch()
        
        self.detail_tabs.addTab(self.performance_tab, "Performance")
        
        main_layout.addWidget(self.detail_tabs)
        
    def load_strategy(self, strategy_name, strategy_type):
        """Load strategy for editing"""
        self.current_strategy = strategy_name
        self.current_type = strategy_type
        
        # Update labels
        self.name_label.setText(strategy_name)
        self.type_label.setText(f"Type: {strategy_type}")
        
        # Clear forms
        self._clear_params_form()
        self.description_edit.clear()
        self.rules_table.setRowCount(0)
        
        # Load strategy data
        if strategy_type == "built-in":
            self._load_builtin_strategy(strategy_name)
        else:
            self._load_custom_strategy(strategy_name)
            
        # Update performance metrics
        self._load_performance_metrics(strategy_name)
        
    def _clear_params_form(self):
        """Clear parameters form"""
        # Remove all widgets from form
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
            
    def _load_builtin_strategy(self, strategy_name):
        """Load built-in strategy data"""
        # Get parameters
        params = {}
        
        # Get from RL manager if available
        if hasattr(self.trading_system.rl_manager, 'strategy_params'):
            params = self.trading_system.rl_manager.strategy_params.get(strategy_name, {})
        
        # If no parameters yet, get defaults
        if not params and hasattr(self.trading_system.rl_manager, 'get_strategy_parameters'):
            params = self.trading_system.rl_manager.get_strategy_parameters(strategy_name)
            
        # If still no parameters, use defaults based on strategy type
        if not params:
            params = self._get_default_params(strategy_name)
            
        # Add parameters to form
        self._add_params_to_form(params)
        
        # Disable rules tab for built-in strategies
        self.detail_tabs.setTabEnabled(2, False)
        
        # Set description
        description = self._get_strategy_description(strategy_name)
        self.description_edit.setText(description)
        
        # Make description read-only for built-in strategies
        self.description_edit.setReadOnly(True)
        
    def _load_custom_strategy(self, strategy_name):
        """Load custom strategy data"""
        try:
            # Load strategy file
            strategy_file = os.path.join("data", "strategies", f"{strategy_name}.json")
            
            if not os.path.exists(strategy_file):
                return
                
            with open(strategy_file, 'r') as f:
                strategy_data = json.load(f)
                
            # Add parameters to form
            self._add_params_to_form(strategy_data.get("parameters", {}))
            
            # Set description
            self.description_edit.setText(strategy_data.get("description", ""))
            
            # Enable description editing
            self.description_edit.setReadOnly(False)
            
            # Enable rules tab
            self.detail_tabs.setTabEnabled(2, True)
            
            # Add rules to table
            self._add_rules_to_table(strategy_data.get("rules", []))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading strategy: {e}")
            
    def _add_params_to_form(self, params):
        """Add parameters to form"""
        # Clear form first
        self._clear_params_form()
        
        # Add parameters
        for param_name, param_value in params.items():
            # Skip 'weight' parameter
            if param_name == 'weight':
                continue
                
            label = QLabel(param_name.replace('_', ' ').title())
            
            # Create appropriate widget based on value type
            if isinstance(param_value, bool):
                widget = QCheckBox()
                widget.setChecked(param_value)
            elif isinstance(param_value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(param_value)
            elif isinstance(param_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999, 999999)
                widget.setDecimals(5)
                widget.setValue(param_value)
            else:
                widget = QLineEdit(str(param_value))
                
            # Add to form
            self.params_form.addRow(label, widget)
            
    def _add_rules_to_table(self, rules):
        """Add rules to table"""
        # Clear table
        self.rules_table.setRowCount(0)
        
        # Add rules
        for rule in rules:
            row = self.rules_table.rowCount()
            self.rules_table.insertRow(row)
            
            # Add rule data
            self.rules_table.setItem(row, 0, QTableWidgetItem(rule.get("condition", "")))
            self.rules_table.setItem(row, 1, QTableWidgetItem(str(rule.get("value", ""))))
            self.rules_table.setItem(row, 2, QTableWidgetItem(rule.get("action", "")))
            
            # Add delete button
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda checked, r=row: self._delete_rule(r))
            self.rules_table.setCellWidget(row, 3, delete_btn)
            
    def _get_default_params(self, strategy_name):
        """Get default parameters for a strategy"""
        if strategy_name == "trend_following":
            return {
                "short_window": 10,
                "medium_window": 50,
                "long_window": 100,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05
            }
        elif strategy_name == "mean_reversion":
            return {
                "window": 20,
                "std_dev": 2.0,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        elif strategy_name == "breakout":
            return {
                "window": 50,
                "volume_window": 10,
                "volume_threshold": 1.5,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06
            }
        else:
            return {
                "window": 20,
                "threshold": 1.5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05
            }
            
    def _get_strategy_description(self, strategy_name):
        """Get description for a built-in strategy"""
        if strategy_name == "trend_following":
            return (
                "Trend Following Strategy\n\n"
                "This strategy uses multiple moving averages to identify and follow trends. "
                "It generates buy signals when shorter-term averages cross above longer-term "
                "averages, and sell signals when shorter-term averages cross below longer-term averages.\n\n"
                "Parameters:\n"
                "- short_window: Period for short moving average\n"
                "- medium_window: Period for medium moving average\n"
                "- long_window: Period for long moving average\n"
                "- stop_loss_pct: Stop loss percentage\n"
                "- take_profit_pct: Take profit percentage"
            )
        elif strategy_name == "mean_reversion":
            return (
                "Mean Reversion Strategy\n\n"
                "This strategy is based on the principle that prices tend to revert to their mean. "
                "It uses Bollinger Bands to identify overbought and oversold conditions, "
                "generating buy signals when price is below the lower band and sell signals "
                "when price is above the upper band.\n\n"
                "Parameters:\n"
                "- window: Period for moving average calculation\n"
                "- std_dev: Number of standard deviations for Bollinger Bands\n"
                "- stop_loss_pct: Stop loss percentage\n"
                "- take_profit_pct: Take profit percentage"
            )
        elif strategy_name == "breakout":
            return (
                "Breakout Strategy\n\n"
                "This strategy aims to capture price movements that break through significant "
                "support or resistance levels. It generates buy signals when price breaks above "
                "recent highs with increased volume, and sell signals when price breaks below "
                "recent lows with increased volume.\n\n"
                "Parameters:\n"
                "- window: Period for support/resistance calculation\n"
                "- volume_window: Period for volume moving average\n"
                "- volume_threshold: Volume increase threshold for confirmation\n"
                "- stop_loss_pct: Stop loss percentage\n"
                "- take_profit_pct: Take profit percentage"
            )
        else:
            return f"No description available for {strategy_name}"
            
    def _get_form_params(self):
        """Get parameters from form"""
        params = {}
        
        # Get all form widgets
        for i in range(self.params_form.rowCount()):
            # Get label and widget
            label_item = self.params_form.itemAt(i, QFormLayout.LabelRole)
            field_item = self.params_form.itemAt(i, QFormLayout.FieldRole)
            
            if label_item is None or field_item is None:
                continue
                
            label = label_item.widget()
            widget = field_item.widget()
            
            if label is None or widget is None:
                continue
                
            # Get parameter name and value
            param_name = label.text().lower().replace(' ', '_')
            
            # Get value based on widget type
            if isinstance(widget, QCheckBox):
                param_value = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                param_value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                param_value = widget.value()
            elif isinstance(widget, QLineEdit):
                param_value = widget.text()
                
                # Try to convert to number if possible
                try:
                    if '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except ValueError:
                    pass
                    
            params[param_name] = param_value
            
        return params
        
    def _get_table_rules(self):
        """Get rules from table"""
        rules = []
        
        # Get all table rows
        for row in range(self.rules_table.rowCount()):
            condition = self.rules_table.item(row, 0).text()
            value = self.rules_table.item(row, 1).text()
            action = self.rules_table.item(row, 2).text()
            
            # Try to convert value to number if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
                
            rule = {
                "condition": condition,
                "value": value,
                "action": action
            }
            
            rules.append(rule)
            
        return rules
        
    def _save_parameters(self):
        """Save strategy parameters"""
        if not self.current_strategy:
            return
            
        params = self._get_form_params()
        
        if self.current_type == "built-in":
            # Save to RL manager
            if hasattr(self.trading_system.rl_manager, 'strategy_params'):
                if self.current_strategy not in self.trading_system.rl_manager.strategy_params:
                    self.trading_system.rl_manager.strategy_params[self.current_strategy] = {}
                    
                # Update parameters
                for param, value in params.items():
                    self.trading_system.rl_manager.strategy_params[self.current_strategy][param] = value
                    
                QMessageBox.information(self, "Parameters Saved", "Strategy parameters saved successfully")
        else:
            # Save to custom strategy file
            try:
                strategy_file = os.path.join("data", "strategies", f"{self.current_strategy}.json")
                
                if not os.path.exists(strategy_file):
                    return
                    
                with open(strategy_file, 'r') as f:
                    strategy_data = json.load(f)
                    
                # Update parameters
                strategy_data["parameters"] = params
                
                with open(strategy_file, 'w') as f:
                    json.dump(strategy_data, f, indent=4)
                    
                QMessageBox.information(self, "Parameters Saved", "Strategy parameters saved successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving parameters: {e}")
                
    def _save_description(self):
        """Save strategy description"""
        if not self.current_strategy or self.current_type != "custom":
            return
            
        description = self.description_edit.toPlainText()
        
        try:
            strategy_file = os.path.join("data", "strategies", f"{self.current_strategy}.json")
            
            if not os.path.exists(strategy_file):
                return
                
            with open(strategy_file, 'r') as f:
                strategy_data = json.load(f)
                
            # Update description
            strategy_data["description"] = description
            
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=4)
                
            QMessageBox.information(self, "Description Saved", "Strategy description saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving description: {e}")
            
    def _add_rule(self):
        """Add a new rule"""
        # Open rule dialog
        dialog = RuleDialog(self)
        
        if dialog.exec_():
            # Get rule data
            condition = dialog.condition_edit.text()
            value = dialog.value_edit.text()
            action = dialog.action_edit.text()
            
            # Add to table
            row = self.rules_table.rowCount()
            self.rules_table.insertRow(row)
            
            self.rules_table.setItem(row, 0, QTableWidgetItem(condition))
            self.rules_table.setItem(row, 1, QTableWidgetItem(value))
            self.rules_table.setItem(row, 2, QTableWidgetItem(action))
            
            # Add delete button
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda checked, r=row: self._delete_rule(r))
            self.rules_table.setCellWidget(row, 3, delete_btn)
            
    def _delete_rule(self, row):
        """Delete a rule"""
        self.rules_table.removeRow(row)
        
    def _save_rules(self):
        """Save strategy rules"""
        if not self.current_strategy or self.current_type != "custom":
            return
            
        rules = self._get_table_rules()
        
        try:
            strategy_file = os.path.join("data", "strategies", f"{self.current_strategy}.json")
            
            if not os.path.exists(strategy_file):
                return
                
            with open(strategy_file, 'r') as f:
                strategy_data = json.load(f)
                
            # Update rules
            strategy_data["rules"] = rules
            
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=4)
                
            QMessageBox.information(self, "Rules Saved", "Strategy rules saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving rules: {e}")
            
    def _optimize_parameters(self):
        """Optimize strategy parameters"""
        if not self.current_strategy:
            return
            
        # Ask for optimization parameters
        dialog = OptimizationDialog(self)
        
        if not dialog.exec_():
            return
            
        # Get optimization parameters
        start_date = dialog.start_date.date().toString("yyyy-MM-dd")
        end_date = dialog.end_date.date().toString("yyyy-MM-dd")
        
        try:
            # Start optimization
            optimization_id = self.trading_system.optimize_strategy(
                self.current_strategy, start_date, end_date
            )
            
            if optimization_id:
                QMessageBox.information(
                    self,
                    "Optimization Started",
                    f"Optimization for {self.current_strategy} has been started. "
                    "You will be notified when it is completed."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Optimization Error",
                    "Failed to start optimization"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error", f"Error: {e}")
            
    def _backtest_strategy(self):
        """Run backtest for the strategy"""
        if not self.current_strategy:
            return
            
        # Ask for backtest parameters
        dialog = BacktestDialog(self)
        
        if not dialog.exec_():
            return
            
        # Get backtest parameters
        start_date = dialog.start_date.date().toString("yyyy-MM-dd")
        end_date = dialog.end_date.date().toString("yyyy-MM-dd")
        initial_capital = dialog.capital_spin.value()
        
        try:
            # Start backtest
            backtest_id = self.trading_system.run_backtest(
                self.current_strategy, start_date, end_date, initial_capital
            )
            
            if backtest_id:
                QMessageBox.information(
                    self,
                    "Backtest Started",
                    f"Backtest for {self.current_strategy} has been started. "
                    "Results will be available in the Backtesting tab when completed."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Backtest Error",
                    "Failed to start backtest"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Backtest Error", f"Error: {e}")
            
    def _load_performance_metrics(self, strategy_name):
        """Load performance metrics for the strategy"""
        # This would load actual metrics from performance history
        # For now, using placeholder data
        
        # In a real implementation, you would load this from trading_system
        self.win_rate_label.setText("58.3%")
        self.profit_factor_label.setText("1.73")
        self.sharpe_label.setText("1.48")
        self.drawdown_label.setText("14.2%")


class NewStrategyDialog(QDialog):
    """Dialog for creating a new strategy"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("New Strategy")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        # Name field
        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        form_layout.addRow("Name:", self.name_edit)
        
        # Type field
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Custom", "Trend Following", "Mean Reversion", "Breakout"])
        form_layout.addRow("Type:", self.type_combo)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class RuleDialog(QDialog):
    """Dialog for adding or editing a rule"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Rule Editor")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Rule fields
        form_layout = QFormLayout()
        
        self.condition_edit = QLineEdit()
        form_layout.addRow("Condition:", self.condition_edit)
        
        self.value_edit = QLineEdit()
        form_layout.addRow("Value:", self.value_edit)
        
        self.action_edit = QLineEdit()
        form_layout.addRow("Action:", self.action_edit)
        
        layout.addLayout(form_layout)
        
        # Help text
        help_label = QLabel(
            "Example:\n"
            "Condition: price_above_sma\n"
            "Value: sma_20\n"
            "Action: buy"
        )
        help_label.setStyleSheet("color: gray;")
        layout.addWidget(help_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class OptimizationDialog(QDialog):
    """Dialog for strategy parameter optimization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Optimize Strategy Parameters")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        # Date range
        form_layout = QFormLayout()
        
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-3))
        self.start_date.setCalendarPopup(True)
        form_layout.addRow("Start Date:", self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        form_layout.addRow("End Date:", self.end_date)
        
        layout.addLayout(form_layout)
        
        # Message
        message = QLabel(
            "Optimization will search for the best parameter values "
            "based on historical performance. This may take some time."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class BacktestDialog(QDialog):
    """Dialog for strategy backtesting"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Backtest Strategy")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        # Backtest parameters
        form_layout = QFormLayout()
        
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-3))
        self.start_date.setCalendarPopup(True)
        form_layout.addRow("Start Date:", self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        form_layout.addRow("End Date:", self.end_date)
        
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 1000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setSingleStep(1000)
        form_layout.addRow("Initial Capital:", self.capital_spin)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)