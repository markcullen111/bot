# backtester_tab.py

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
                           QPushButton, QComboBox, QDateEdit, QDoubleSpinBox, QTabWidget,
                           QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, 
                           QCheckBox, QSpinBox, QProgressBar, QMessageBox, QSplitter,
                           QFrame, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QColor

class BacktesterTab(QWidget):
    """Backtesting interface for evaluating trading strategies"""
    
    backtest_started = pyqtSignal(str)  # Signal emitted when backtest starts
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.current_backtest_id = None
        self.backtest_results = {}
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        
        # Top section: Configuration
        config_group = QGroupBox("Backtest Configuration")
        config_layout = QFormLayout(config_group)
        
        # Strategy selection
        self.strategy_combo = QComboBox()
        self._populate_strategies()
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        config_layout.addRow("Strategy:", self.strategy_combo)
        
        # Date range selection
        date_layout = QHBoxLayout()
        
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-3))
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.end_date)
        
        config_layout.addRow("Date Range:", date_layout)
        
        # Initial capital
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setSingleStep(1000)
        self.initial_capital.setPrefix("$")
        config_layout.addRow("Initial Capital:", self.initial_capital)
        
        # Trading fee
        self.trading_fee = QDoubleSpinBox()
        self.trading_fee.setRange(0, 1)
        self.trading_fee.setValue(0.001)  # 0.1% default
        self.trading_fee.setSingleStep(0.001)
        self.trading_fee.setDecimals(4)
        self.trading_fee.setSuffix("%")
        self.trading_fee.setSpecialValueText("No Fee")
        config_layout.addRow("Trading Fee:", self.trading_fee)
        
        # Parameters accordion (expanded when strategy selected)
        self.params_group = QGroupBox("Strategy Parameters")
        self.params_group.setCheckable(True)
        self.params_group.setChecked(False)
        self.params_layout = QFormLayout(self.params_group)
        config_layout.addRow(self.params_group)
        
        # Advanced options
        self.advanced_group = QGroupBox("Advanced Options")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QFormLayout(self.advanced_group)
        
        self.use_ai_models = QCheckBox("Use AI Models")
        self.use_ai_models.setChecked(True)
        advanced_layout.addRow("", self.use_ai_models)
        
        self.optimize_params = QCheckBox("Optimize Parameters")
        advanced_layout.addRow("", self.optimize_params)
        
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 9999)
        self.random_seed.setValue(42)
        advanced_layout.addRow("Random Seed:", self.random_seed)
        
        config_layout.addRow(self.advanced_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self._run_backtest)
        button_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_backtest)
        button_layout.addWidget(self.stop_btn)
        
        self.optimize_btn = QPushButton("Optimize Strategy")
        self.optimize_btn.clicked.connect(self._optimize_strategy)
        button_layout.addWidget(self.optimize_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_results)
        button_layout.addWidget(self.save_btn)
        
        config_layout.addRow("", button_layout)
        
        # Progress indicator
        self.progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)
        
        config_layout.addRow("Status:", self.progress_layout)
        
        main_layout.addWidget(config_group)
        
        # Bottom section: Results tabs
        self.results_tabs = QTabWidget()
        
        # Performance tab
        self.performance_tab = QWidget()
        performance_layout = QVBoxLayout(self.performance_tab)
        
        # Chart for equity curve
        self.equity_chart = pg.PlotWidget()
        self.equity_chart.setBackground('w')
        self.equity_chart.setLabel('left', 'Portfolio Value')
        self.equity_chart.setLabel('bottom', 'Date')
        self.equity_chart.showGrid(x=True, y=True)
        
        # Create legend
        self.equity_chart.addLegend()
        
        performance_layout.addWidget(self.equity_chart)
        
        # Performance metrics table
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        # Create two-column layout for metrics
        metrics_columns = QHBoxLayout()
        
        # Left column
        left_metrics = QFormLayout()
        self.roi_label = QLabel("0.00%")
        left_metrics.addRow("Return on Investment:", self.roi_label)
        
        self.sharpe_label = QLabel("0.00")
        left_metrics.addRow("Sharpe Ratio:", self.sharpe_label)
        
        self.drawdown_label = QLabel("0.00%")
        left_metrics.addRow("Max Drawdown:", self.drawdown_label)
        
        metrics_columns.addLayout(left_metrics)
        
        # Right column
        right_metrics = QFormLayout()
        self.win_rate_label = QLabel("0.00%")
        right_metrics.addRow("Win Rate:", self.win_rate_label)
        
        self.profit_factor_label = QLabel("0.00")
        right_metrics.addRow("Profit Factor:", self.profit_factor_label)
        
        self.trades_label = QLabel("0")
        right_metrics.addRow("Total Trades:", self.trades_label)
        
        metrics_columns.addLayout(right_metrics)
        
        metrics_layout.addRow(metrics_columns)
        performance_layout.addWidget(metrics_group)
        
        self.results_tabs.addTab(self.performance_tab, "Performance")
        
        # Trades tab
        self.trades_tab = QWidget()
        trades_layout = QVBoxLayout(self.trades_tab)
        
        self.trades_table = QTableWidget(0, 8)
        self.trades_table.setHorizontalHeaderLabels([
            "Entry Date", "Exit Date", "Symbol", "Direction", 
            "Entry Price", "Exit Price", "P&L ($)", "P&L (%)"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
        
        self.results_tabs.addTab(self.trades_tab, "Trades")
        
        # Comparison tab
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)
        
        # Add chart for comparing multiple strategies
        self.comparison_chart = pg.PlotWidget()
        self.comparison_chart.setBackground('w')
        self.comparison_chart.setLabel('left', 'Portfolio Value')
        self.comparison_chart.setLabel('bottom', 'Date')
        self.comparison_chart.showGrid(x=True, y=True)
        self.comparison_chart.addLegend()
        
        comparison_layout.addWidget(self.comparison_chart)
        
        # Add comparison metrics table
        self.comparison_table = QTableWidget(0, 7)
        self.comparison_table.setHorizontalHeaderLabels([
            "Strategy", "ROI", "Sharpe", "Max DD", "Win Rate", 
            "Profit Factor", "Trades"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        comparison_layout.addWidget(self.comparison_table)
        
        self.results_tabs.addTab(self.comparison_tab, "Comparison")
        
        main_layout.addWidget(self.results_tabs)
        
    def _populate_strategies(self):
        """Populate strategy dropdown with available strategies"""
        # Clear existing items
        self.strategy_combo.clear()
        
        # Add strategies from trading system
        if hasattr(self.trading_system, 'strategy_system') and \
           hasattr(self.trading_system.strategy_system, 'strategy_generator') and \
           hasattr(self.trading_system.strategy_system.strategy_generator, 'strategies'):
            
            strategies = list(self.trading_system.strategy_system.strategy_generator.strategies.keys())
            self.strategy_combo.addItems(strategies)
            
        # Add custom strategies if available
        custom_strategies_dir = os.path.join("data", "strategies")
        if os.path.exists(custom_strategies_dir):
            for filename in os.listdir(custom_strategies_dir):
                if filename.endswith(".json"):
                    strategy_name = filename[:-5]  # Remove .json extension
                    # Add if not already in the list
                    if self.strategy_combo.findText(strategy_name) == -1:
                        self.strategy_combo.addItem(strategy_name)
    
    def _on_strategy_changed(self, index):
        """Handle strategy selection change"""
        if index < 0:
            return
            
        strategy_name = self.strategy_combo.currentText()
        self._load_strategy_parameters(strategy_name)
        
    def _load_strategy_parameters(self, strategy_name):
        """Load parameters for selected strategy"""
        # Clear existing parameters
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
            
        # Get strategy parameters
        params = self._get_strategy_parameters(strategy_name)
        
        if not params:
            self.params_group.setChecked(False)
            return
            
        # Add parameters to form
        for param_name, param_value in params.items():
            # Skip non-numeric parameters
            if not isinstance(param_value, (int, float)):
                continue
                
            label = QLabel(param_name.replace('_', ' ').title())
            
            # Create appropriate editor based on parameter type
            if isinstance(param_value, int):
                editor = QSpinBox()
                editor.setRange(1, 1000)
                editor.setValue(param_value)
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(4)
                editor.setRange(0, 100)
                editor.setSingleStep(0.1)
                editor.setValue(param_value)
                
            self.params_layout.addRow(label, editor)
            
        # Show parameters group
        self.params_group.setChecked(True)
        
    def _get_strategy_parameters(self, strategy_name):
        """Get parameters for a strategy"""
        # Check if parameters exist in RL manager
        if hasattr(self.trading_system, 'rl_manager') and \
           hasattr(self.trading_system.rl_manager, 'strategy_params') and \
           strategy_name in self.trading_system.rl_manager.strategy_params:
            return self.trading_system.rl_manager.strategy_params[strategy_name]
            
        # Check custom strategy file
        custom_strategy_path = os.path.join("data", "strategies", f"{strategy_name}.json")
        if os.path.exists(custom_strategy_path):
            try:
                with open(custom_strategy_path, 'r') as f:
                    strategy_data = json.load(f)
                return strategy_data.get("parameters", {})
            except:
                pass
                
        # Return default parameters based on strategy type
        if strategy_name == "trend_following":
            return {"short_window": 10, "medium_window": 50, "long_window": 100}
        elif strategy_name == "mean_reversion":
            return {"window": 20, "std_dev": 2.0}
        elif strategy_name == "breakout":
            return {"window": 50, "volume_window": 10, "volume_threshold": 1.5}
        
        return {}
        
    def _get_parameter_values(self):
        """Get current parameter values from UI"""
        params = {}
        
        for i in range(self.params_layout.rowCount()):
            label_item = self.params_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.params_layout.itemAt(i, QFormLayout.FieldRole)
            
            if not label_item or not field_item:
                continue
                
            label = label_item.widget()
            editor = field_item.widget()
            
            if not label or not editor:
                continue
                
            param_name = label.text().replace(' ', '_').lower()
            
            if isinstance(editor, QSpinBox) or isinstance(editor, QDoubleSpinBox):
                param_value = editor.value()
                params[param_name] = param_value
                
        return params
        
    def _run_backtest(self):
        """Run backtest with current configuration"""
        # Get configuration
        strategy_name = self.strategy_combo.currentText()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        initial_capital = self.initial_capital.value()
        trading_fee = self.trading_fee.value() / 100  # Convert from percentage
        parameters = self._get_parameter_values()
        use_ai = self.use_ai_models.isChecked()
        
        # Input validation
        if not strategy_name:
            QMessageBox.warning(self, "Configuration Error", "Please select a strategy.")
            return
        
        # Update UI
        self.progress_label.setText("Initializing...")
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create backtest task
        config = {
            "strategy": strategy_name,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "trading_fee": trading_fee,
            "parameters": parameters,
            "use_ai_models": use_ai,
            "random_seed": self.random_seed.value()
        }
        
        # Submit backtest task to thread manager if available
        if hasattr(self.trading_system, 'thread_manager'):
            self.current_backtest_id = self.trading_system.thread_manager.submit_task(
                f"backtest_{strategy_name}",
                self._run_backtest_task,
                args=(config,),
                priority=2  # Medium priority
            )
            
            # Connect to progress updates
            self.trading_system.thread_manager.task_progress.connect(self._update_backtest_progress)
            self.trading_system.thread_manager.task_completed.connect(self._on_backtest_completed)
        else:
            # Run directly if thread manager not available
            self.progress_label.setText("Running backtest (synchronous)...")
            result = self._run_backtest_task(config)
            self._process_backtest_results(result)
            self._update_backtest_progress("backtest", 100)
            self._update_ui_after_backtest()
        
    def _run_backtest_task(self, config, progress_callback=None):
        """Execute backtest (runs in worker thread)"""
        try:
            # Update progress
            if progress_callback:
                progress_callback(10)
                
            # Simulate backtest execution or call actual backtest engine
            if hasattr(self.trading_system, 'run_backtest'):
                # Use actual backtest engine if available
                results = self.trading_system.run_backtest(
                    config["strategy"],
                    config["start_date"],
                    config["end_date"],
                    config["initial_capital"],
                    config["parameters"],
                    config["trading_fee"],
                    config["use_ai_models"]
                )
            else:
                # Simulate backtest results for demonstration
                results = self._simulate_backtest_results(config)
                
            # Update progress
            if progress_callback:
                progress_callback(100)
                
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
            
    def _update_backtest_progress(self, task_id, progress):
        """Update backtest progress bar"""
        if task_id == self.current_backtest_id:
            self.progress_bar.setValue(progress)
            
            if progress < 100:
                self.progress_label.setText(f"Running backtest ({progress}%)...")
            else:
                self.progress_label.setText("Backtest completed")
                
    def _on_backtest_completed(self, task_result):
        """Handle backtest completion"""
        if task_result.task_id != self.current_backtest_id:
            return
            
        if task_result.success:
            # Process results
            self._process_backtest_results(task_result.result)
        else:
            # Handle error
            QMessageBox.critical(self, "Backtest Error", f"Error running backtest: {task_result.error}")
            self.progress_label.setText(f"Error: {task_result.error}")
            
        # Update UI
        self._update_ui_after_backtest()
        
    def _update_ui_after_backtest(self):
        """Update UI after backtest completion"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
    def _process_backtest_results(self, results):
        """Process and display backtest results"""
        if not results or "error" in results:
            return
            
        # Store results
        strategy_name = self.strategy_combo.currentText()
        self.backtest_results[strategy_name] = results
        
        # Update equity curve chart
        self._update_equity_curve(results)
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        # Update trades table
        self._update_trades_table(results)
        
        # Update comparison chart and table
        self._update_comparison_view()
        
    def _update_equity_curve(self, results):
        """Update equity curve chart with backtest results"""
        self.equity_chart.clear()
        
        if "equity_curve" not in results:
            return
            
        equity_curve = results["equity_curve"]
        dates = range(len(equity_curve))  # Use indices for x-axis
        
        # Plot equity curve
        pen = pg.mkPen(color='b', width=2)
        self.equity_chart.plot(dates, equity_curve, name="Portfolio Value", pen=pen)
        
        # Add benchmark if available
        if "benchmark" in results:
            benchmark = results["benchmark"]
            pen = pg.mkPen(color='r', width=1, style=Qt.DashLine)
            self.equity_chart.plot(dates, benchmark, name="Benchmark", pen=pen)
            
    def _update_performance_metrics(self, results):
        """Update performance metrics with backtest results"""
        if "metrics" not in results:
            return
            
        metrics = results["metrics"]
        
        # Update labels
        self.roi_label.setText(f"{metrics.get('roi', 0) * 100:.2f}%")
        self.sharpe_label.setText(f"{metrics.get('sharpe_ratio', 0):.2f}")
        self.drawdown_label.setText(f"{metrics.get('max_drawdown', 0) * 100:.2f}%")
        self.win_rate_label.setText(f"{metrics.get('win_rate', 0) * 100:.2f}%")
        self.profit_factor_label.setText(f"{metrics.get('profit_factor', 0):.2f}")
        self.trades_label.setText(f"{metrics.get('total_trades', 0)}")
        
    def _update_trades_table(self, results):
        """Update trades table with backtest results"""
        self.trades_table.setRowCount(0)
        
        if "trades" not in results:
            return
            
        trades = results["trades"]
        
        # Add trades to table
        for i, trade in enumerate(trades):
            self.trades_table.insertRow(i)
            
            # Format dates
            entry_date = trade.get("entry_time", "")
            if isinstance(entry_date, (float, int)):
                entry_date = datetime.fromtimestamp(entry_date).strftime("%Y-%m-%d %H:%M")
                
            exit_date = trade.get("exit_time", "")
            if isinstance(exit_date, (float, int)):
                exit_date = datetime.fromtimestamp(exit_date).strftime("%Y-%m-%d %H:%M")
            
            # Calculate P&L percentage
            pnl = trade.get("pnl", 0)
            entry_price = trade.get("entry_price", 0)
            pnl_pct = (pnl / (entry_price * trade.get("amount", 0))) * 100 if entry_price else 0
            
            # Add data to row
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(entry_date)))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(exit_date)))
            self.trades_table.setItem(i, 2, QTableWidgetItem(trade.get("symbol", "")))
            self.trades_table.setItem(i, 3, QTableWidgetItem(trade.get("action", "")))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('entry_price', 0):.6f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('exit_price', 0):.6f}"))
            
            # Apply color to P&L cells
            pnl_item = QTableWidgetItem(f"{pnl:.2f}")
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
            
            if pnl > 0:
                pnl_item.setForeground(QColor('green'))
                pnl_pct_item.setForeground(QColor('green'))
            elif pnl < 0:
                pnl_item.setForeground(QColor('red'))
                pnl_pct_item.setForeground(QColor('red'))
                
            self.trades_table.setItem(i, 6, pnl_item)
            self.trades_table.setItem(i, 7, pnl_pct_item)
        
    def _update_comparison_view(self):
        """Update comparison chart and table with all backtest results"""
        # Clear existing
        self.comparison_chart.clear()
        self.comparison_table.setRowCount(0)
        
        # Plot equity curves for all results
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        color_index = 0
        
        for i, (strategy_name, results) in enumerate(self.backtest_results.items()):
            if "equity_curve" not in results or "metrics" not in results:
                continue
                
            # Add to chart
            equity_curve = results["equity_curve"]
            dates = range(len(equity_curve))
            
            color = colors[color_index % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            self.comparison_chart.plot(dates, equity_curve, name=strategy_name, pen=pen)
            
            color_index += 1
            
            # Add to table
            metrics = results["metrics"]
            self.comparison_table.insertRow(i)
            
            self.comparison_table.setItem(i, 0, QTableWidgetItem(strategy_name))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{metrics.get('roi', 0) * 100:.2f}%"))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{metrics.get('sharpe_ratio', 0):.2f}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{metrics.get('max_drawdown', 0) * 100:.2f}%"))
            self.comparison_table.setItem(i, 4, QTableWidgetItem(f"{metrics.get('win_rate', 0) * 100:.2f}%"))
            self.comparison_table.setItem(i, 5, QTableWidgetItem(f"{metrics.get('profit_factor', 0):.2f}"))
            self.comparison_table.setItem(i, 6, QTableWidgetItem(f"{metrics.get('total_trades', 0)}"))
            
    def _stop_backtest(self):
        """Stop current backtest"""
        if self.current_backtest_id and hasattr(self.trading_system, 'thread_manager'):
            self.trading_system.thread_manager.cancel_task(self.current_backtest_id)
            self.progress_label.setText("Backtest cancelled")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
    def _optimize_strategy(self):
        """Optimize strategy parameters"""
        # Get configuration
        strategy_name = self.strategy_combo.currentText()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        
        # Input validation
        if not strategy_name:
            QMessageBox.warning(self, "Configuration Error", "Please select a strategy.")
            return
            
        # Show optimization dialog
        parameters = self._get_parameter_values()
        
        # Execute optimization if available
        if hasattr(self.trading_system, 'optimize_strategy'):
            # Update UI
            self.progress_label.setText("Optimizing parameters...")
            self.progress_bar.setValue(0)
            self.run_btn.setEnabled(False)
            self.optimize_btn.setEnabled(False)
            
            # Start optimization task
            if hasattr(self.trading_system, 'thread_manager'):
                task_id = self.trading_system.thread_manager.submit_task(
                    f"optimize_{strategy_name}",
                    self._run_optimization_task,
                    args=(strategy_name, start_date, end_date, parameters),
                    priority=3  # Lower priority than backtest
                )
                
                # Connect to progress updates
                self.trading_system.thread_manager.task_progress.connect(
                    lambda task_id, progress: self._update_optimization_progress(task_id, progress))
                self.trading_system.thread_manager.task_completed.connect(
                    lambda result: self._on_optimization_completed(result))
            else:
                # Run directly
                best_params = self._run_optimization_task(strategy_name, start_date, end_date, parameters)
                self._apply_optimized_parameters(strategy_name, best_params)
                
                self.progress_label.setText("Optimization completed")
                self.run_btn.setEnabled(True)
                self.optimize_btn.setEnabled(True)
        else:
            QMessageBox.information(self, "Not Available", 
                                  "Strategy optimization is not available in this version.")
    
    def _run_optimization_task(self, strategy_name, start_date, end_date, initial_params, progress_callback=None):
        """Execute optimization task (runs in worker thread)"""
        # Call trading system's optimize_strategy method if available
        if hasattr(self.trading_system, 'optimize_strategy'):
            return self.trading_system.optimize_strategy(
                strategy_name, start_date, end_date, initial_params, progress_callback)
        return None
        
    def _update_optimization_progress(self, task_id, progress):
        """Update optimization progress"""
        if task_id.startswith("optimize_"):
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Optimizing parameters ({progress}%)...")
            
    def _on_optimization_completed(self, task_result):
        """Handle optimization completion"""
        if not task_result.task_id.startswith("optimize_"):
            return
            
        if task_result.success and task_result.result:
            # Apply optimized parameters
            strategy_name = task_result.task_id.split("_", 1)[1]
            self._apply_optimized_parameters(strategy_name, task_result.result)
            
            QMessageBox.information(self, "Optimization Complete", 
                                  "Parameter optimization completed successfully.")
        else:
            QMessageBox.warning(self, "Optimization Error", 
                               f"Error optimizing parameters: {task_result.error}")
            
        # Update UI
        self.progress_label.setText("Ready")
        self.run_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        
    def _apply_optimized_parameters(self, strategy_name, optimized_params):
        """Apply optimized parameters to UI and strategy"""
        if not optimized_params:
            return
            
        # Update strategy parameters in trading system if needed
        if hasattr(self.trading_system, 'rl_manager') and \
           hasattr(self.trading_system.rl_manager, 'strategy_params'):
            if strategy_name not in self.trading_system.rl_manager.strategy_params:
                self.trading_system.rl_manager.strategy_params[strategy_name] = {}
                
            self.trading_system.rl_manager.strategy_params[strategy_name].update(optimized_params)
            
        # Reload parameters in UI
        self._load_strategy_parameters(strategy_name)
        
    def _save_results(self):
        """Save backtest results to file"""
        if not self.backtest_results:
            return
            
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Backtest Results", "", "CSV Files (*.csv);;JSON Files (*.json)")
            
        if not file_path:
            return
            
        strategy_name = self.strategy_combo.currentText()
        results = self.backtest_results.get(strategy_name)
        
        if not results:
            return
            
        # Save based on file type
        if file_path.endswith(".json"):
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
        else:
            # Default to CSV
            if "trades" in results:
                # Save trades as CSV
                trades_df = pd.DataFrame(results["trades"])
                trades_df.to_csv(file_path, index=False)
                
        QMessageBox.information(self, "Export Complete", f"Results saved to {file_path}")
        
    def _simulate_backtest_results(self, config):
        """Simulate backtest results for demonstration"""
        # Create sample equity curve
        days = 90
        initial_capital = config["initial_capital"]
        np.random.seed(config["random_seed"])
        
        # Generate random equity curve with trend
        daily_returns = np.random.normal(0.001, 0.02, days)  # Mean 0.1%, std 2%
        
        if config["strategy"] == "trend_following":
            # Add trend bias
            daily_returns += 0.002  # 0.2% daily edge
        elif config["strategy"] == "mean_reversion":
            # Add mean reversion pattern
            daily_returns = np.sin(np.linspace(0, 6*np.pi, days)) * 0.02 + np.random.normal(0.001, 0.01, days)
            
        equity_curve = initial_capital * np.cumprod(1 + daily_returns)
        
        # Generate benchmark (buy and hold)
        benchmark_returns = np.random.normal(0.0005, 0.015, days)  # Mean 0.05%, std 1.5%
        benchmark = initial_capital * np.cumprod(1 + benchmark_returns)
        
        # Generate sample trades
        num_trades = int(days / 3)  # Average hold time of 3 days
        trades = []
        
        for i in range(num_trades):
            # Random entry date
            entry_day = np.random.randint(0, days - 5)
            # Random hold time 1-10 days
            hold_time = np.random.randint(1, 10)
            exit_day = min(entry_day + hold_time, days - 1)
            
            # Entry and exit prices
            entry_price = equity_curve[entry_day] / 100  # Arbitrary scaling
            exit_price = equity_curve[exit_day] / 100
            
            # Trade direction
            action = "buy" if np.random.random() > 0.3 else "sell"
            
            # Adjust exit price based on direction
            if action == "sell":
                pnl = entry_price - exit_price
            else:
                pnl = exit_price - entry_price
                
            # Adjust for random win/loss
            if np.random.random() > 0.6:  # 60% win rate
                pnl = abs(pnl)
            else:
                pnl = -abs(pnl) * 0.8  # Smaller losses
                
            # Create trade record
            amount = initial_capital * 0.1 / entry_price  # 10% of capital
            
            trade = {
                "entry_time": datetime.now() - timedelta(days=days-entry_day),
                "exit_time": datetime.now() - timedelta(days=days-exit_day),
                "symbol": config["strategy"][:3].upper() + "/USDT",
                "action": action,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": amount,
                "pnl": pnl * amount
            }
            
            trades.append(trade)
            
        # Calculate metrics
        total_pnl = sum(t["pnl"] for t in trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades)) if sum(t["pnl"] for t in losing_trades) else float('inf')
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        # Calculate Sharpe ratio
        roi = (equity_curve[-1] / equity_curve[0]) - 1
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        metrics = {
            "roi": roi,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": num_trades,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }
        
        return {
            "equity_curve": equity_curve.tolist(),
            "benchmark": benchmark.tolist(),
            "trades": trades,
            "metrics": metrics,
            "config": config
        }
        
    def refresh(self):
        """Refresh backtester tab data"""
        self._populate_strategies()
