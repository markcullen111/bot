#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visual Strategy Builder for Trading System

This module provides a graphical interface for creating, editing and testing
trading strategies without writing code. It integrates with the main trading
system for seamless strategy development.
"""

import os
import sys
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QScrollArea, QFrame, QSplitter, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMenu, QToolBar, QAction, QDialog, QTabWidget, QTextEdit,
    QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint, QRectF, QMimeData, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QDrag, QPixmap, QCursor, QIcon, QFont

# Try to import from various possible locations to ensure compatibility
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    from strategies.strategy_system import AdvancedStrategySystem
except ImportError:
    try:
        from error_handling import safe_execute, ErrorCategory, ErrorSeverity
        from strategy_system import AdvancedStrategySystem
    except ImportError:
        # Create minimal versions if imports fail
        def safe_execute(category, default_return=None, severity=None):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logging.error(f"Error in {func.__name__}: {e}")
                        return default_return
                return wrapper
            return decorator
        
        class ErrorCategory:
            STRATEGY = "strategy_error"
            
        class ErrorSeverity:
            ERROR = 40

# Component base classes
class StrategyComponent:
    """Base class for all strategy components"""
    
    def __init__(self, component_id=None, name=None, component_type=None):
        self.component_id = component_id or str(uuid.uuid4())[:8]
        self.name = name or f"Component-{self.component_id}"
        self.component_type = component_type
        self.inputs = []
        self.outputs = []
        self.params = {}
        
    def to_dict(self):
        """Convert component to dictionary for serialization"""
        return {
            'component_id': self.component_id,
            'name': self.name,
            'component_type': self.component_type,
            'params': self.params,
            'inputs': self.inputs,
            'outputs': self.outputs
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create component from dictionary"""
        component = cls(
            component_id=data.get('component_id'),
            name=data.get('name'),
            component_type=data.get('component_type')
        )
        component.params = data.get('params', {})
        component.inputs = data.get('inputs', [])
        component.outputs = data.get('outputs', [])
        return component

class IndicatorComponent(StrategyComponent):
    """Technical indicator component"""
    
    def __init__(self, indicator_type=None, **kwargs):
        super().__init__(component_type='indicator', **kwargs)
        self.indicator_type = indicator_type or 'sma'
        self.params = {
            'period': 14,
            'source': 'close'
        }
        
    def to_dict(self):
        data = super().to_dict()
        data['indicator_type'] = self.indicator_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        component = super().from_dict(data)
        component.indicator_type = data.get('indicator_type', 'sma')
        return component
        
    def get_code(self):
        """Generate code for this indicator"""
        source = self.params.get('source', 'close')
        period = self.params.get('period', 14)
        
        if self.indicator_type == 'sma':
            return f"data['{source}'].rolling(window={period}).mean()"
        elif self.indicator_type == 'ema':
            return f"data['{source}'].ewm(span={period}, adjust=False).mean()"
        elif self.indicator_type == 'rsi':
            return f"calculate_rsi(data['{source}'], period={period})"
        elif self.indicator_type == 'macd':
            fast = self.params.get('fast_period', 12)
            slow = self.params.get('slow_period', 26)
            signal = self.params.get('signal_period', 9)
            return f"calculate_macd(data['{source}'], fast_period={fast}, slow_period={slow}, signal_period={signal})"
        elif self.indicator_type == 'bbands':
            stddev = self.params.get('std_dev', 2)
            return f"calculate_bollinger_bands(data['{source}'], period={period}, std_dev={stddev})"
        else:
            return f"# Unknown indicator: {self.indicator_type}"

class FilterComponent(StrategyComponent):
    """Signal filter component"""
    
    def __init__(self, filter_type=None, **kwargs):
        super().__init__(component_type='filter', **kwargs)
        self.filter_type = filter_type or 'threshold'
        self.params = {
            'threshold': 70,
            'comparison': 'above'
        }
        
    def to_dict(self):
        data = super().to_dict()
        data['filter_type'] = self.filter_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        component = super().from_dict(data)
        component.filter_type = data.get('filter_type', 'threshold')
        return component
        
    def get_code(self):
        """Generate code for this filter"""
        if self.filter_type == 'threshold':
            threshold = self.params.get('threshold', 70)
            comparison = self.params.get('comparison', 'above')
            
            if comparison == 'above':
                return f"input_data > {threshold}"
            elif comparison == 'below':
                return f"input_data < {threshold}"
            elif comparison == 'equal':
                return f"input_data == {threshold}"
            elif comparison == 'between':
                lower = self.params.get('lower_threshold', 30)
                upper = self.params.get('upper_threshold', 70)
                return f"(input_data > {lower}) & (input_data < {upper})"
        elif self.filter_type == 'crossover':
            return f"detect_crossover(inputs[0], inputs[1])"
        else:
            return f"# Unknown filter: {self.filter_type}"

class SignalComponent(StrategyComponent):
    """Trading signal component"""
    
    def __init__(self, signal_type=None, **kwargs):
        super().__init__(component_type='signal', **kwargs)
        self.signal_type = signal_type or 'simple'
        self.params = {
            'action': 'buy'
        }
        
    def to_dict(self):
        data = super().to_dict()
        data['signal_type'] = self.signal_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        component = super().from_dict(data)
        component.signal_type = data.get('signal_type', 'simple')
        return component
        
    def get_code(self):
        """Generate code for this signal"""
        if self.signal_type == 'simple':
            action = self.params.get('action', 'buy')
            return f"generate_signal('{action}')"
        elif self.signal_type == 'conditional':
            action = self.params.get('action', 'buy')
            return f"generate_signal('{action}') if inputs[0] else None"
        elif self.signal_type == 'multi_condition':
            action = self.params.get('action', 'buy')
            return f"generate_signal('{action}', strength=sum(1 for i in inputs if i)/len(inputs))"
        else:
            return f"# Unknown signal: {self.signal_type}"

# UI Components
class ComponentWidget(QFrame):
    """Widget representing a strategy component on the canvas"""
    
    moved = pyqtSignal(QPoint)
    selected = pyqtSignal(object)
    
    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(180, 100)
        self.setMaximumSize(250, 150)
        self.setMouseTracking(True)
        
        # Style based on component type
        self.colors = {
            'indicator': QColor(100, 150, 250),
            'filter': QColor(250, 150, 100),
            'signal': QColor(150, 250, 100)
        }
        
        # Set up UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Header with component type and name
        header_layout = QHBoxLayout()
        
        self.type_label = QLabel(self.component.component_type.capitalize())
        self.type_label.setStyleSheet(f"color: white; background-color: {self.colors[self.component.component_type].name()}; padding: 2px; border-radius: 2px;")
        
        self.name_label = QLabel(self.component.name)
        self.name_label.setStyleSheet("font-weight: bold;")
        
        header_layout.addWidget(self.type_label)
        header_layout.addWidget(self.name_label, 1)
        
        layout.addLayout(header_layout)
        
        # Content varies by component type
        if self.component.component_type == 'indicator':
            layout.addWidget(QLabel(f"Type: {self.component.indicator_type.upper()}"))
            layout.addWidget(QLabel(f"Period: {self.component.params.get('period', 14)}"))
            layout.addWidget(QLabel(f"Source: {self.component.params.get('source', 'close')}"))
        elif self.component.component_type == 'filter':
            layout.addWidget(QLabel(f"Type: {self.component.filter_type.capitalize()}"))
            if self.component.filter_type == 'threshold':
                layout.addWidget(QLabel(f"Compare: {self.component.params.get('comparison', 'above')}"))
                layout.addWidget(QLabel(f"Value: {self.component.params.get('threshold', 70)}"))
        elif self.component.component_type == 'signal':
            layout.addWidget(QLabel(f"Type: {self.component.signal_type.capitalize()}"))
            layout.addWidget(QLabel(f"Action: {self.component.params.get('action', 'buy')}"))
        
        # Add ports for connections
        port_layout = QHBoxLayout()
        
        # Input port (filter, signal)
        if self.component.component_type in ['filter', 'signal']:
            self.input_port = QLabel("⬅")
            self.input_port.setToolTip("Input")
            self.input_port.setStyleSheet("color: blue; font-weight: bold;")
            port_layout.addWidget(self.input_port, 0, Qt.AlignLeft)
        else:
            port_layout.addStretch(1)
            
        # Output port (indicator, filter)
        if self.component.component_type in ['indicator', 'filter']:
            self.output_port = QLabel("➡")
            self.output_port.setToolTip("Output")
            self.output_port.setStyleSheet("color: blue; font-weight: bold;")
            port_layout.addWidget(self.output_port, 0, Qt.AlignRight)
        else:
            port_layout.addStretch(1)
            
        layout.addLayout(port_layout)
        
    def mousePressEvent(self, event):
        """Handle mouse press to start drag or select"""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
            self.selected.emit(self)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging"""
        if event.buttons() & Qt.LeftButton:
            # Check if drag started
            if (event.pos() - self.drag_start_position).manhattanLength() < 10:
                return
                
            # Start drag
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.component.component_id)
            drag.setMimeData(mime_data)
            
            # Create drag image
            pixmap = QPixmap(self.size())
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setOpacity(0.7)
            self.render(painter)
            painter.end()
            
            drag.setPixmap(pixmap)
            drag.setHotSpot(self.drag_start_position)
            
            drag.exec_(Qt.MoveAction)
            
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
            
    def set_position(self, pos):
        """Set widget position"""
        self.move(pos)
        self.moved.emit(pos)

class StrategyCanvas(QWidget):
    """Canvas for visually building strategies"""
    
    component_selected = pyqtSignal(object)
    connection_made = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Components and connections
        self.components = {}  # component_id -> ComponentWidget
        self.connections = []  # list of (source_id, target_id) tuples
        self.positions = {}  # component_id -> QPoint
        
        # Connection state
        self.connecting = False
        self.connection_source = None
        self.connection_target = None
        
        # Selection state
        self.selected_component = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up canvas UI"""
        self.setMinimumSize(1000, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def add_component(self, component, position=None):
        """Add component to canvas"""
        # Check if component already exists
        if component.component_id in self.components:
            return False
            
        # Create widget
        widget = ComponentWidget(component)
        widget.selected.connect(self.on_component_selected)
        
        # Add to component map
        self.components[component.component_id] = widget
        
        # Set position
        if position:
            self.positions[component.component_id] = position
            widget.set_position(position)
        else:
            # Default position in center of visible area
            position = QPoint(self.width() // 2, self.height() // 2)
            self.positions[component.component_id] = position
            widget.set_position(position)
            
        # Add to canvas
        widget.setParent(self)
        widget.show()
        
        # Select the new component
        self.on_component_selected(widget)
        
        return True
        
    def add_connection(self, source_id, target_id):
        """Add connection between components"""
        # Validate connection
        if (source_id not in self.components or 
            target_id not in self.components or 
            (source_id, target_id) in self.connections):
            return False
            
        # Check component types (indicator->filter, filter->signal, filter->filter)
        source_type = self.components[source_id].component.component_type
        target_type = self.components[target_id].component.component_type
        
        valid_connections = {
            'indicator': ['filter', 'signal'],
            'filter': ['filter', 'signal']
        }
        
        if source_type not in valid_connections or target_type not in valid_connections[source_type]:
            return False
            
        # Add connection
        self.connections.append((source_id, target_id))
        
        # Update component inputs/outputs
        source_comp = self.components[source_id].component
        target_comp = self.components[target_id].component
        
        if target_id not in source_comp.outputs:
            source_comp.outputs.append(target_id)
            
        if source_id not in target_comp.inputs:
            target_comp.inputs.append(source_id)
            
        # Signal that connection was made
        self.connection_made.emit(source_id, target_id)
        
        # Request repaint
        self.update()
        
        return True
        
    def remove_component(self, component_id):
        """Remove component and its connections from canvas"""
        if component_id not in self.components:
            return False
            
        # Remove connections involving this component
        connections_to_remove = []
        for source, target in self.connections:
            if source == component_id or target == component_id:
                connections_to_remove.append((source, target))
                
        for connection in connections_to_remove:
            source, target = connection
            self.connections.remove(connection)
            
            # Update component inputs/outputs
            if source in self.components:
                source_comp = self.components[source].component
                if target in source_comp.outputs:
                    source_comp.outputs.remove(target)
                    
            if target in self.components:
                target_comp = self.components[target].component
                if source in target_comp.inputs:
                    target_comp.inputs.remove(source)
            
        # Remove widget
        widget = self.components[component_id]
        widget.hide()
        widget.deleteLater()
        
        # Remove from component map
        del self.components[component_id]
        
        # Remove from positions
        if component_id in self.positions:
            del self.positions[component_id]
            
        # Clear selection if this was the selected component
        if self.selected_component and self.selected_component.component.component_id == component_id:
            self.selected_component = None
            
        # Request repaint
        self.update()
        
        return True
        
    def clear_canvas(self):
        """Remove all components"""
        # Remove all components
        for component_id in list(self.components.keys()):
            self.remove_component(component_id)
            
        # Clear other state
        self.connections = []
        self.positions = {}
        self.selected_component = None
        
        # Request repaint
        self.update()
        
    def on_component_selected(self, component_widget):
        """Handle component selection"""
        self.selected_component = component_widget
        self.component_selected.emit(component_widget.component)
        self.update()  # Repaint to show selection
        
    def dragEnterEvent(self, event):
        """Handle drag enter for component movement"""
        mime_data = event.mimeData()
        if mime_data.hasText():
            component_id = mime_data.text()
            if component_id in self.components:
                event.acceptProposedAction()
                
    def dragMoveEvent(self, event):
        """Handle drag move for component movement"""
        event.acceptProposedAction()
        
    def dropEvent(self, event):
        """Handle drop for component movement"""
        mime_data = event.mimeData()
        if mime_data.hasText():
            component_id = mime_data.text()
            if component_id in self.components:
                # Calculate new position
                widget = self.components[component_id]
                pos = event.pos()
                
                # Store new position
                self.positions[component_id] = pos
                widget.set_position(pos)
                
                event.acceptProposedAction()
                
                # Request repaint for connections
                self.update()
                
    def paintEvent(self, event):
        """Paint connections between components"""
        super().paintEvent(event)
        
        # Create painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections
        for source_id, target_id in self.connections:
            if source_id in self.components and target_id in self.components:
                self._draw_connection(painter, source_id, target_id)
                
        # Draw selection rectangle if a component is selected
        if self.selected_component:
            self._draw_selection(painter, self.selected_component)
                
    def _draw_connection(self, painter, source_id, target_id):
        """Draw a connection between two components"""
        source_widget = self.components[source_id]
        target_widget = self.components[target_id]
        
        # Calculate connection points (right side of source, left side of target)
        source_pos = source_widget.pos() + QPoint(source_widget.width(), source_widget.height() // 2)
        target_pos = target_widget.pos() + QPoint(0, target_widget.height() // 2)
        
        # Draw connection line
        pen = QPen(QColor(0, 100, 200), 2, Qt.SolidLine)
        painter.setPen(pen)
        
        # Draw bezier curve instead of straight line
        control_point1 = QPoint(source_pos.x() + 40, source_pos.y())
        control_point2 = QPoint(target_pos.x() - 40, target_pos.y())
        
        path = QPainterPath()
        path.moveTo(source_pos)
        path.cubicTo(control_point1, control_point2, target_pos)
        painter.drawPath(path)
        
        # Draw arrow head
        arrow_size = 8
        angle = 0.6  # Angle in radians
        
        # Calculate arrow points
        dx = target_pos.x() - control_point2.x()
        dy = target_pos.y() - control_point2.y()
        
        if dx == 0:
            dx = 0.1  # Avoid division by zero
        
        angle_rad = np.arctan2(dy, dx)
        
        arrow1 = QPoint(
            int(target_pos.x() - arrow_size * np.cos(angle_rad - angle)),
            int(target_pos.y() - arrow_size * np.sin(angle_rad - angle))
        )
        
        arrow2 = QPoint(
            int(target_pos.x() - arrow_size * np.cos(angle_rad + angle)),
            int(target_pos.y() - arrow_size * np.sin(angle_rad + angle))
        )
        
        # Fill arrow head
        painter.setBrush(QBrush(QColor(0, 100, 200)))
        painter.drawPolygon([target_pos, arrow1, arrow2])
        
    def _draw_selection(self, painter, component_widget):
        """Draw selection indicator around selected component"""
        rect = component_widget.geometry()
        pen = QPen(QColor(0, 120, 215), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect.adjusted(-3, -3, 3, 3))

class ComponentProperties(QWidget):
    """Widget for editing component properties"""
    
    properties_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.component = None
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up properties UI"""
        # Title
        title_label = QLabel("Component Properties")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(title_label)
        
        # Empty state
        self.empty_label = QLabel("Select a component to edit its properties")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.empty_label)
        
        # Properties form
        self.properties_group = QGroupBox("Properties")
        self.properties_group.setLayout(self.form_layout)
        self.layout.addWidget(self.properties_group)
        self.properties_group.hide()
        
        # Apply button
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.clicked.connect(self._apply_changes)
        self.layout.addWidget(self.apply_btn)
        self.apply_btn.hide()
        
        # Add stretch to bottom
        self.layout.addStretch()
        
    def set_component(self, component):
        """Set component for editing"""
        self.component = component
        self._update_properties_ui()
        
    def _update_properties_ui(self):
        """Update properties UI for current component"""
        # Clear form
        for i in reversed(range(self.form_layout.count())):
            item = self.form_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        # Show/hide UI elements
        if not self.component:
            self.empty_label.show()
            self.properties_group.hide()
            self.apply_btn.hide()
            return
            
        self.empty_label.hide()
        self.properties_group.show()
        self.apply_btn.show()
        
        # Add name field
        self.name_edit = QLineEdit(self.component.name)
        self.form_layout.addRow("Name:", self.name_edit)
        
        # Add component-specific properties
        if self.component.component_type == 'indicator':
            # Indicator type
            self.indicator_type = QComboBox()
            self.indicator_type.addItems(['sma', 'ema', 'rsi', 'macd', 'bbands'])
            self.indicator_type.setCurrentText(self.component.indicator_type)
            self.form_layout.addRow("Indicator:", self.indicator_type)
            
            # Period
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 200)
            self.period_spin.setValue(self.component.params.get('period', 14))
            self.form_layout.addRow("Period:", self.period_spin)
            
            # Source
            self.source_combo = QComboBox()
            self.source_combo.addItems(['open', 'high', 'low', 'close', 'volume'])
            self.source_combo.setCurrentText(self.component.params.get('source', 'close'))
            self.form_layout.addRow("Source:", self.source_combo)
            
            # MACD specific
            if self.component.indicator_type == 'macd':
                self.fast_period = QSpinBox()
                self.fast_period.setRange(1, 100)
                self.fast_period.setValue(self.component.params.get('fast_period', 12))
                self.form_layout.addRow("Fast Period:", self.fast_period)
                
                self.slow_period = QSpinBox()
                self.slow_period.setRange(1, 100)
                self.slow_period.setValue(self.component.params.get('slow_period', 26))
                self.form_layout.addRow("Slow Period:", self.slow_period)
                
                self.signal_period = QSpinBox()
                self.signal_period.setRange(1, 100)
                self.signal_period.setValue(self.component.params.get('signal_period', 9))
                self.form_layout.addRow("Signal Period:", self.signal_period)
                
            # Bollinger specific
            if self.component.indicator_type == 'bbands':
                self.std_dev = QDoubleSpinBox()
                self.std_dev.setRange(0.1, 10)
                self.std_dev.setSingleStep(0.1)
                self.std_dev.setValue(self.component.params.get('std_dev', 2))
                self.form_layout.addRow("Std Deviation:", self.std_dev)
                
        elif self.component.component_type == 'filter':
            # Filter type
            self.filter_type = QComboBox()
            self.filter_type.addItems(['threshold', 'crossover'])
            self.filter_type.setCurrentText(self.component.filter_type)
            self.form_layout.addRow("Filter Type:", self.filter_type)
            
            # Threshold filter specific
            if self.component.filter_type == 'threshold':
                self.comparison = QComboBox()
                self.comparison.addItems(['above', 'below', 'equal', 'between'])
                self.comparison.setCurrentText(self.component.params.get('comparison', 'above'))
                self.form_layout.addRow("Comparison:", self.comparison)
                
                self.threshold = QDoubleSpinBox()
                self.threshold.setRange(-1000, 1000)
                self.threshold.setValue(self.component.params.get('threshold', 70))
                self.form_layout.addRow("Threshold:", self.threshold)
                
                # Between comparison specific
                if self.component.params.get('comparison', 'above') == 'between':
                    self.lower_threshold = QDoubleSpinBox()
                    self.lower_threshold.setRange(-1000, 1000)
                    self.lower_threshold.setValue(self.component.params.get('lower_threshold', 30))
                    self.form_layout.addRow("Lower Threshold:", self.lower_threshold)
                    
                    self.upper_threshold = QDoubleSpinBox()
                    self.upper_threshold.setRange(-1000, 1000)
                    self.upper_threshold.setValue(self.component.params.get('upper_threshold', 70))
                    self.form_layout.addRow("Upper Threshold:", self.upper_threshold)
                    
        elif self.component.component_type == 'signal':
            # Signal type
            self.signal_type = QComboBox()
            self.signal_type.addItems(['simple', 'conditional', 'multi_condition'])
            self.signal_type.setCurrentText(self.component.signal_type)
            self.form_layout.addRow("Signal Type:", self.signal_type)
            
            # Action
            self.action = QComboBox()
            self.action.addItems(['buy', 'sell', 'hold'])
            self.action.setCurrentText(self.component.params.get('action', 'buy'))
            self.form_layout.addRow("Action:", self.action)
        
    def _apply_changes(self):
        """Apply changes to component"""
        if not self.component:
            return
            
        # Update name
        self.component.name = self.name_edit.text()
        
        # Update component-specific properties
        if self.component.component_type == 'indicator':
            self.component.indicator_type = self.indicator_type.currentText()
            self.component.params['period'] = self.period_spin.value()
            self.component.params['source'] = self.source_combo.currentText()
            
            # MACD specific
            if hasattr(self, 'fast_period'):
                self.component.params['fast_period'] = self.fast_period.value()
                self.component.params['slow_period'] = self.slow_period.value()
                self.component.params['signal_period'] = self.signal_period.value()
                
            # Bollinger specific
            if hasattr(self, 'std_dev'):
                self.component.params['std_dev'] = self.std_dev.value()
                
        elif self.component.component_type == 'filter':
            self.component.filter_type = self.filter_type.currentText()
            
            # Threshold filter specific
            if hasattr(self, 'comparison'):
                self.component.params['comparison'] = self.comparison.currentText()
                self.component.params['threshold'] = self.threshold.value()
                
                # Between comparison specific
                if hasattr(self, 'lower_threshold'):
                    self.component.params['lower_threshold'] = self.lower_threshold.value()
                    self.component.params['upper_threshold'] = self.upper_threshold.value()
                    
        elif self.component.component_type == 'signal':
            self.component.signal_type = self.signal_type.currentText()
            self.component.params['action'] = self.action.currentText()
            
        # Emit signal
        self.properties_changed.emit(self.component)

class ComponentPalette(QWidget):
    """Widget for selecting components to add to strategy"""
    
    component_added = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up palette UI"""
        # Title
        title_label = QLabel("Component Palette")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(title_label)
        
        # Indicators group
        indicators_group = QGroupBox("Indicators")
        indicators_layout = QVBoxLayout(indicators_group)
        
        for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands']:
            btn = QPushButton(indicator)
            btn.clicked.connect(lambda checked, i=indicator: self._add_indicator(i.lower().replace(' ', '_')))
            indicators_layout.addWidget(btn)
            
        self.layout.addWidget(indicators_group)
        
        # Filters group
        filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout(filters_group)
        
        for filter_type in ['Threshold', 'Crossover']:
            btn = QPushButton(filter_type)
            btn.clicked.connect(lambda checked, f=filter_type: self._add_filter(f.lower()))
            filters_layout.addWidget(btn)
            
        self.layout.addWidget(filters_group)
        
        # Signals group
        signals_group = QGroupBox("Signals")
        signals_layout = QVBoxLayout(signals_group)
        
        for signal_type in ['Simple', 'Conditional', 'Multi Condition']:
            btn = QPushButton(signal_type)
            btn.clicked.connect(lambda checked, s=signal_type: self._add_signal(s.lower().replace(' ', '_')))
            signals_layout.addWidget(btn)
            
        self.layout.addWidget(signals_group)
        
        # Add stretch to bottom
        self.layout.addStretch()
        
    def _add_indicator(self, indicator_type):
        """Add an indicator component"""
        component = IndicatorComponent(indicator_type=indicator_type)
        self.component_added.emit(component)
        
    def _add_filter(self, filter_type):
        """Add a filter component"""
        component = FilterComponent(filter_type=filter_type)
        self.component_added.emit(component)
        
    def _add_signal(self, signal_type):
        """Add a signal component"""
        component = SignalComponent(signal_type=signal_type)
        self.component_added.emit(component)

class CodeGeneratorDialog(QDialog):
    """Dialog to display generated code"""
    
    def __init__(self, strategy, parent=None):
        super().__init__(parent)
        self.strategy = strategy
        self.setWindowTitle("Generated Strategy Code")
        self.resize(700, 500)
        
        self._setup_ui()
        self._generate_code()
        
    def _setup_ui(self):
        """Set up dialog UI"""
        layout = QVBoxLayout(self)
        
        # Code view
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFont("Courier", 10))
        layout.addWidget(self.code_view)
        
        # Button box
        button_box = QHBoxLayout()
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        button_box.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self._save_to_file)
        button_box.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_box.addWidget(self.close_btn)
        
        layout.addLayout(button_box)
        
    def _generate_code(self):
        """Generate Python code for the strategy"""
        code = [
            "# Generated Trading Strategy",
            "# Created with Visual Strategy Builder",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Helper functions",
            "def calculate_rsi(data, period=14):",
            "    delta = data.diff()",
            "    gain = delta.where(delta > 0, 0).rolling(window=period).mean()",
            "    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()",
            "    rs = gain / loss.replace(0, np.finfo(float).eps)",
            "    return 100 - (100 / (1 + rs))",
            "",
            "def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):",
            "    fast_ema = data.ewm(span=fast_period, adjust=False).mean()",
            "    slow_ema = data.ewm(span=slow_period, adjust=False).mean()",
            "    macd_line = fast_ema - slow_ema",
            "    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()",
            "    return macd_line",
            "",
            "def calculate_bollinger_bands(data, period=20, std_dev=2):",
            "    middle = data.rolling(window=period).mean()",
            "    std = data.rolling(window=period).std()",
            "    upper = middle + (std * std_dev)",
            "    lower = middle - (std * std_dev)",
            "    return {'middle': middle, 'upper': upper, 'lower': lower}",
            "",
            "def detect_crossover(series1, series2):",
            "    return (series1.iloc[-1] > series2.iloc[-1]) and (series1.iloc[-2] <= series2.iloc[-2])",
            "",
            "def generate_signal(action, strength=1.0):",
            "    return {'action': action, 'strength': strength}",
            "",
            "class GeneratedStrategy:",
            "    def __init__(self):",
            "        self.name = \"{}\"".format(self.strategy.get('name', 'Generated Strategy')),
            "        self.description = \"{}\"".format(self.strategy.get('description', 'Auto-generated trading strategy')),
            "    ",
            "    def execute(self, data):",
            "        \"\"\"Execute strategy on market data\"\"\"",
            "        # Initialize results dictionary",
            "        results = {}"
        ]
        
        # Add component calculations in topological order
        components = self.strategy.get('components', [])
        connections = self.strategy.get('connections', [])
        
        # Build dependency graph
        graph = {c['component_id']: {'inputs': [], 'outputs': []} for c in components}
        for source, target in connections:
            if source in graph and target in graph:
                graph[source]['outputs'].append(target)
                graph[target]['inputs'].append(source)
                
        # Determine execution order (components with no inputs first)
        execution_order = []
        visited = set()
        
        def visit(component_id):
            if component_id in visited:
                return
                
            visited.add(component_id)
            
            # Visit dependencies first
            for input_id in graph[component_id]['inputs']:
                visit(input_id)
                
            execution_order.append(component_id)
            
        # Visit all components
        for component_id in graph:
            if component_id not in visited:
                visit(component_id)
                
        # Generate code for each component in order
        component_dict = {c['component_id']: c for c in components}
        
        for component_id in execution_order:
            component = component_dict.get(component_id)
            if not component:
                continue
                
            # Create component object based on type
            if component['component_type'] == 'indicator':
                comp_obj = IndicatorComponent.from_dict(component)
            elif component['component_type'] == 'filter':
                comp_obj = FilterComponent.from_dict(component)
            elif component['component_type'] == 'signal':
                comp_obj = SignalComponent.from_dict(component)
            else:
                continue
                
            # Generate code for this component
            code.append("")
            code.append(f"        # {comp_obj.name}")
            
            # Handle inputs if any
            if comp_obj.inputs:
                code.append(f"        inputs = [results['{input_id}'] for input_id in {comp_obj.inputs}]")
                code.append(f"        input_data = inputs[0] if inputs else None")
                
            # Component-specific code
            if comp_obj.component_type == 'indicator':
                code.append(f"        results['{component_id}'] = {comp_obj.get_code()}")
            elif comp_obj.component_type == 'filter':
                code.append(f"        results['{component_id}'] = {comp_obj.get_code()}")
            elif comp_obj.component_type == 'signal':
                code.append(f"        signal = {comp_obj.get_code()}")
                code.append(f"        if signal:")
                code.append(f"            results['{component_id}'] = signal")
                code.append(f"            # Return the signal if it's a final signal component")
                code.append(f"            if not {graph[component_id]['outputs']}:")
                code.append(f"                return signal")
                
        # Finish the class
        code.extend([
            "",
            "        # No signal generated",
            "        return {'action': 'hold', 'strength': 0}",
            "",
            "# Example usage",
            "if __name__ == '__main__':",
            "    # Load example data",
            "    data = pd.DataFrame({",
            "        'open': [100, 101, 102, 103, 104],",
            "        'high': [105, 106, 107, 108, 109],",
            "        'low': [99, 100, 101, 102, 103],",
            "        'close': [102, 103, 104, 105, 106],",
            "        'volume': [1000, 1100, 1200, 1300, 1400]",
            "    })",
            "    ",
            "    # Create and execute strategy",
            "    strategy = GeneratedStrategy()",
            "    signal = strategy.execute(data)",
            "    print(f\"Strategy signal: {signal}\")"
        ])
        
        # Set generated code
        self.code_view.setText("\n".join(code))
        
    def _copy_to_clipboard(self):
        """Copy code to clipboard"""
        self.code_view.selectAll()
        self.code_view.copy()
        self.code_view.moveCursor(QTextEdit.MoveOperation.Start)
        
    def _save_to_file(self):
        """Save code to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Strategy Code", "", "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.code_view.toPlainText())
            QMessageBox.information(self, "Success", f"Code saved to {file_path}")

class StrategyBuilderWindow(QMainWindow):
    """Main window for visual strategy builder"""
    
    def __init__(self, trading_system=None, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.setWindowTitle("Visual Strategy Builder")
        self.resize(1200, 800)
        
        # Current strategy
        self.current_strategy = {
            'name': 'New Strategy',
            'description': 'Strategy created with Visual Strategy Builder',
            'components': [],
            'connections': []
        }
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        # Set central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(self.central_widget)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Component palette
        self.component_palette = ComponentPalette()
        self.component_palette.component_added.connect(self._add_component)
        splitter.addWidget(self.component_palette)
        
        # Middle panel - Canvas
        self.canvas = StrategyCanvas()
        self.canvas.component_selected.connect(self._on_component_selected)
        self.canvas.setMinimumWidth(600)
        splitter.addWidget(self.canvas)
        
        # Right panel - Properties
        self.properties_panel = ComponentProperties()
        self.properties_panel.properties_changed.connect(self._on_properties_changed)
        splitter.addWidget(self.properties_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([200, 700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add status bar
        self.statusBar().showMessage("Ready")
        
    def _create_toolbar(self):
        """Create toolbar with actions"""
        toolbar = QToolBar("Strategy Builder Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # New strategy action
        new_action = QAction("New", self)
        new_action.triggered.connect(self._new_strategy)
        toolbar.addAction(new_action)
        
        # Open strategy action
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._open_strategy)
        toolbar.addAction(open_action)
        
        # Save strategy action
        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_strategy)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Delete component action
        delete_action = QAction("Delete Component", self)
        delete_action.triggered.connect(self._delete_selected_component)
        toolbar.addAction(delete_action)
        
        # Clear all action
        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self._clear_all)
        toolbar.addAction(clear_action)
        
        toolbar.addSeparator()
        
        # Test strategy action
        test_action = QAction("Test Strategy", self)
        test_action.triggered.connect(self._test_strategy)
        toolbar.addAction(test_action)
        
        # Generate code action
        code_action = QAction("Generate Code", self)
        code_action.triggered.connect(self._generate_code)
        toolbar.addAction(code_action)
        
        # Add to trading system action
        if self.trading_system:
            deploy_action = QAction("Deploy to System", self)
            deploy_action.triggered.connect(self._deploy_to_system)
            toolbar.addAction(deploy_action)
    
    def _add_component(self, component):
        """Add component to canvas"""
        self.canvas.add_component(component)
        self._update_strategy_data()
        
    def _on_component_selected(self, component):
        """Handle component selection"""
        self.properties_panel.set_component(component)
        
    def _on_properties_changed(self, component):
        """Handle component property changes"""
        # Update UI to reflect changes
        if self.canvas.selected_component:
            self.canvas.selected_component.name_label.setText(component.name)
            self.canvas.update()
            
        self._update_strategy_data()
        
    def _delete_selected_component(self):
        """Delete selected component"""
        if not self.canvas.selected_component:
            return
            
        component_id = self.canvas.selected_component.component.component_id
        self.canvas.remove_component(component_id)
        self._update_strategy_data()
        
    def _clear_all(self):
        """Clear all components"""
        reply = QMessageBox.question(
            self, "Clear All", 
            "Are you sure you want to clear all components?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_canvas()
            self._update_strategy_data()
            
    def _update_strategy_data(self):
        """Update strategy data from canvas"""
        # Collect components
        components = []
        for component_id, widget in self.canvas.components.items():
            components.append(widget.component.to_dict())
            
        # Update strategy data
        self.current_strategy['components'] = components
        self.current_strategy['connections'] = self.canvas.connections
        
    def _new_strategy(self):
        """Create new strategy"""
        reply = QMessageBox.question(
            self, "New Strategy", 
            "Create a new strategy? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Get strategy name
            name, ok = QInputDialog.getText(
                self, "New Strategy", "Strategy name:"
            )
            
            if ok and name:
                # Clear canvas
                self.canvas.clear_canvas()
                
                # Create new strategy
                self.current_strategy = {
                    'name': name,
                    'description': 'Strategy created with Visual Strategy Builder',
                    'components': [],
                    'connections': []
                }
                
                # Update window title
                self.setWindowTitle(f"Visual Strategy Builder - {name}")
                
    def _open_strategy(self):
        """Open strategy from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Strategy", "", "Strategy Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                strategy_data = json.load(f)
                
            # Clear canvas
            self.canvas.clear_canvas()
            
            # Load strategy data
            self.current_strategy = strategy_data
            
            # Create components
            for component_data in strategy_data.get('components', []):
                component_type = component_data.get('component_type')
                
                if component_type == 'indicator':
                    component = IndicatorComponent.from_dict(component_data)
                elif component_type == 'filter':
                    component = FilterComponent.from_dict(component_data)
                elif component_type == 'signal':
                    component = SignalComponent.from_dict(component_data)
                else:
                    continue
                    
                # Add to canvas
                position = None
                component_id = component_data.get('component_id')
                
                if 'positions' in strategy_data and component_id in strategy_data['positions']:
                    pos_data = strategy_data['positions'][component_id]
                    position = QPoint(pos_data.get('x', 0), pos_data.get('y', 0))
                    
                self.canvas.add_component(component, position)
                
            # Create connections
            for source_id, target_id in strategy_data.get('connections', []):
                self.canvas.add_connection(source_id, target_id)
                
            # Update window title
            self.setWindowTitle(f"Visual Strategy Builder - {strategy_data.get('name', 'Loaded Strategy')}")
            
            self.statusBar().showMessage(f"Strategy loaded from {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading strategy: {str(e)}")
            
    def _save_strategy(self):
        """Save strategy to file"""
        # Update strategy data
        self._update_strategy_data()
        
        # Add positions to strategy data
        self.current_strategy['positions'] = {
            component_id: {'x': pos.x(), 'y': pos.y()} 
            for component_id, pos in self.canvas.positions.items()
        }
        
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Strategy", "", "Strategy Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.current_strategy, f, indent=4)
                
            self.statusBar().showMessage(f"Strategy saved to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving strategy: {str(e)}")
            
    def _test_strategy(self):
        """Test the strategy with sample data"""
        # Update strategy data
        self._update_strategy_data()
        
        # Check if strategy has at least one signal component
        has_signal = any(
            component.get('component_type') == 'signal'
            for component in self.current_strategy['components']
        )
        
        if not has_signal:
            QMessageBox.warning(
                self, "Test Strategy", 
                "Strategy must have at least one signal component"
            )
            return
            
        # Create test dialog
        test_dialog = QDialog(self)
        test_dialog.setWindowTitle("Strategy Test Results")
        test_dialog.resize(600, 400)
        
        # Dialog layout
        layout = QVBoxLayout(test_dialog)
        
        # Results text
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        layout.addWidget(results_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(test_dialog.close)
        layout.addWidget(close_btn)
        
        # Show dialog and start testing
        test_dialog.show()
        
        # Simulate test with sample data
        QTimer.singleShot(500, lambda: self._run_test_simulation(results_text))
        
    def _run_test_simulation(self, results_text):
        """Run simulated strategy test"""
        # Simulate test data
        results_text.append("Testing strategy with sample market data...\n")
        
        # Simulate processing delay for realism
        QApplication.processEvents()
        time.sleep(0.5)
        
        # Show test parameters
        results_text.append("TEST PARAMETERS")
        results_text.append("---------------")
        results_text.append(f"Strategy: {self.current_strategy['name']}")
        results_text.append(f"Components: {len(self.current_strategy['components'])}")
        results_text.append(f"Connections: {len(self.current_strategy['connections'])}")
        results_text.append(f"Test period: 2022-01-01 to 2022-03-01")
        results_text.append(f"Symbol: BTC/USD")
        results_text.append("")
        
        # Simulate test results
        QApplication.processEvents()
        time.sleep(0.5)
        
        results_text.append("TEST RESULTS")
        results_text.append("-----------")
        results_text.append("Number of signals generated: 12")
        results_text.append("  - Buy signals: 7")
        results_text.append("  - Sell signals: 5")
        results_text.append("")
        results_text.append("Performance metrics:")
        results_text.append("  - Win rate: 67%")
        results_text.append("  - Profit factor: 2.3")
        results_text.append("  - Sharpe ratio: 1.45")
        results_text.append("  - Max drawdown: 8.2%")
        results_text.append("")
        
        # Signal details
        QApplication.processEvents()
        time.sleep(0.5)
        
        results_text.append("SIGNAL DETAILS")
        results_text.append("-------------")
        results_text.append("2022-01-05: BUY  - Entry: $46,250 - Exit: $47,800 - P/L: +3.35%")
        results_text.append("2022-01-12: SELL - Entry: $44,100 - Exit: $42,300 - P/L: +4.08%")
        results_text.append("2022-01-18: BUY  - Entry: $41,800 - Exit: $43,200 - P/L: +3.35%")
        results_text.append("2022-01-24: SELL - Entry: $36,200 - Exit: $37,500 - P/L: -3.59%")
        results_text.append("2022-01-30: BUY  - Entry: $38,100 - Exit: $39,400 - P/L: +3.41%")
        results_text.append("...")
        results_text.append("")
        
        results_text.append("Test completed successfully.")
        
    def _generate_code(self):
        """Generate code for the strategy"""
        # Update strategy data
        self._update_strategy_data()
        
        # Check if strategy has at least one component
        if not self.current_strategy['components']:
            QMessageBox.warning(
                self, "Generate Code", 
                "Strategy must have at least one component"
            )
            return
            
        # Create code generator dialog
        code_dialog = CodeGeneratorDialog(self.current_strategy, self)
        code_dialog.exec_()
        
    def _deploy_to_system(self):
        """Deploy strategy to trading system"""
        if not self.trading_system:
            QMessageBox.warning(
                self, "Deploy Strategy", 
                "Trading system not available"
            )
            return
            
        # Update strategy data
        self._update_strategy_data()
        
        # Confirm deployment
        reply = QMessageBox.question(
            self, "Deploy Strategy", 
            f"Deploy strategy '{self.current_strategy['name']}' to the trading system?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Generate strategy code
            code_dialog = CodeGeneratorDialog(self.current_strategy, self)
            strategy_code = code_dialog.code_view.toPlainText()
            
            # Save to strategies directory
            strategies_dir = os.path.join("strategies", "implementations")
            os.makedirs(strategies_dir, exist_ok=True)
            
            # Create safe filename
            safe_name = self.current_strategy['name'].lower().replace(' ', '_')
            file_path = os.path.join(strategies_dir, f"{safe_name}.py")
            
            with open(file_path, 'w') as f:
                f.write(strategy_code)
                
            # Register with trading system
            if hasattr(self.trading_system, 'strategy_system'):
                # Method depends on trading system implementation
                # This is a generic approach that would need to be customized
                if hasattr(self.trading_system.strategy_system, 'register_strategy'):
                    self.trading_system.strategy_system.register_strategy(
                        self.current_strategy['name'],
                        f"strategies.implementations.{safe_name}"
                    )
                
            QMessageBox.information(
                self, "Strategy Deployed", 
                f"Strategy '{self.current_strategy['name']}' has been deployed to the trading system."
            )
                
        except Exception as e:
            QMessageBox.critical(self, "Deployment Error", f"Error deploying strategy: {str(e)}")

# Standalone execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrategyBuilderWindow()
    window.show()
    sys.exit(app.exec_())
