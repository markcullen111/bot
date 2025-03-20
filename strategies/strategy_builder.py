# strategy_builder

import json
import uuid
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFrame, QScrollArea,
    QTabWidget, QGroupBox, QCheckBox, QMenu, QInputDialog, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QSplitter, QDockWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QMimeData, QSize
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QColor, QIcon, QCursor

class StrategyComponent:
    """Base class for strategy components"""
    
    def __init__(self, component_id=None, component_type=None, name=None, params=None):
        self.component_id = component_id or str(uuid.uuid4())
        self.component_type = component_type
        self.name = name or f"Component {self.component_id[:6]}"
        self.params = params or {}
        self.inputs = []
        self.outputs = []
        
    def to_dict(self):
        """Convert component to dictionary"""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'name': self.name,
            'params': self.params,
            'inputs': self.inputs,
            'outputs': self.outputs
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create component from dictionary"""
        component = cls(
            component_id=data.get('component_id'),
            component_type=data.get('component_type'),
            name=data.get('name'),
            params=data.get('params', {})
        )
        component.inputs = data.get('inputs', [])
        component.outputs = data.get('outputs', [])
        return component
        
    def execute(self, inputs):
        """Execute component with inputs"""
        # Base implementation - subclasses should override
        return {}

class IndicatorComponent(StrategyComponent):
    """Component for technical indicators"""
    
    def __init__(self, indicator_type=None, **kwargs):
        super().__init__(component_type='indicator', **kwargs)
        self.indicator_type = indicator_type or 'sma'
        self.params.setdefault('period', 14)
        self.params.setdefault('source', 'close')
        
    def to_dict(self):
        """Convert indicator to dictionary"""
        data = super().to_dict()
        data['indicator_type'] = self.indicator_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        """Create indicator from dictionary"""
        indicator = super().from_dict(data)
        indicator.indicator_type = data.get('indicator_type', 'sma')
        return indicator
        
    def execute(self, data):
        """Calculate indicator value"""
        # Get source data
        source = self.params.get('source', 'close')
        period = self.params.get('period', 14)
        
        if source not in data:
            logging.error(f"Source {source} not found in data")
            return None
            
        # Calculate indicator based on type
        if self.indicator_type == 'sma':
            return data[source].rolling(window=period).mean()
        elif self.indicator_type == 'ema':
            return data[source].ewm(span=period, adjust=False).mean()
        elif self.indicator_type == 'rsi':
            delta = data[source].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        elif self.indicator_type == 'macd':
            # MACD line
            ema12 = data[source].ewm(span=12, adjust=False).mean()
            ema26 = data[source].ewm(span=26, adjust=False).mean()
            return ema12 - ema26
        elif self.indicator_type == 'bbands':
            # Bollinger Bands
            middle = data[source].rolling(window=period).mean()
            std_dev = data[source].rolling(window=period).std()
            upper = middle + (std_dev * self.params.get('std_dev', 2))
            lower = middle - (std_dev * self.params.get('std_dev', 2))
            return {'middle': middle, 'upper': upper, 'lower': lower}
        else:
            logging.error(f"Unknown indicator type: {self.indicator_type}")
            return None

class FilterComponent(StrategyComponent):
    """Component for signal filters"""
    
    def __init__(self, filter_type=None, **kwargs):
        super().__init__(component_type='filter', **kwargs)
        self.filter_type = filter_type or 'threshold'
        self.params.setdefault('threshold', 70)
        self.params.setdefault('comparison', 'above')
        
    def to_dict(self):
        """Convert filter to dictionary"""
        data = super().to_dict()
        data['filter_type'] = self.filter_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        """Create filter from dictionary"""
        filter_comp = super().from_dict(data)
        filter_comp.filter_type = data.get('filter_type', 'threshold')
        return filter_comp
        
    def execute(self, data):
        """Apply filter to data"""
        if self.filter_type == 'threshold':
            threshold = self.params.get('threshold', 70)
            comparison = self.params.get('comparison', 'above')
            
            if comparison == 'above':
                return data > threshold
            elif comparison == 'below':
                return data < threshold
            elif comparison == 'equal':
                return data == threshold
            elif comparison == 'between':
                lower = self.params.get('lower_threshold', 30)
                upper = self.params.get('upper_threshold', 70)
                return (data > lower) & (data < upper)
        elif self.filter_type == 'crossover':
            # Check for crossover between two inputs
            if len(self.inputs) < 2:
                logging.error("Crossover filter requires two inputs")
                return None
                
            # Get previous values for comparison
            current = data[self.inputs[0]].iloc[-1] > data[self.inputs[1]].iloc[-1]
            previous = data[self.inputs[0]].iloc[-2] <= data[self.inputs[1]].iloc[-2]
            
            # Detect crossover
            return current and previous
        else:
            logging.error(f"Unknown filter type: {self.filter_type}")
            return None

class SignalComponent(StrategyComponent):
    """Component for generating trading signals"""
    
    def __init__(self, signal_type=None, **kwargs):
        super().__init__(component_type='signal', **kwargs)
        self.signal_type = signal_type or 'simple'
        self.params.setdefault('action', 'buy')
        
    def to_dict(self):
        """Convert signal to dictionary"""
        data = super().to_dict()
        data['signal_type'] = self.signal_type
        return data
        
    @classmethod
    def from_dict(cls, data):
        """Create signal from dictionary"""
        signal = super().from_dict(data)
        signal.signal_type = data.get('signal_type', 'simple')
        return signal
        
    def execute(self, inputs):
        """Generate trading signal"""
        if self.signal_type == 'simple':
            action = self.params.get('action', 'buy')
            return {'action': action, 'strength': 1.0}
        elif self.signal_type == 'conditional':
            # Check condition - assume inputs[0] is a boolean condition
            if not inputs[0]:
                return None
                
            action = self.params.get('action', 'buy')
            return {'action': action, 'strength': 1.0}
        elif self.signal_type == 'multi_condition':
            # Multiple conditions with weights
            conditions_met = sum(1 for condition in inputs if condition)
            total_conditions = len(inputs)
            
            if conditions_met == 0:
                return None
                
            # Calculate signal strength based on conditions met
            strength = conditions_met / total_conditions
            action = self.params.get('action', 'buy')
            
            return {'action': action, 'strength': strength}
        else:
            logging.error(f"Unknown signal type: {self.signal_type}")
            return None

class ComponentWidget(QFrame):
    """Widget representing a strategy component"""
    
    moved = pyqtSignal(QPoint)
    selected = pyqtSignal(object)
    connected = pyqtSignal(str, str)
    
    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(200, 120)
        self.setMaximumSize(300, 200)
        self.setMouseTracking(True)
        
        # Set colors based on component type
        self.colors = {
            'indicator': QColor(120, 180, 255),
            'filter': QColor(255, 180, 120),
            'signal': QColor(180, 255, 120)
        }
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up widget UI"""
        layout = QVBoxLayout(self)
        
        # Header with component type and name
        header_layout = QHBoxLayout()
        self.type_label = QLabel(self.component.component_type.capitalize())
        self.type_label.setStyleSheet(f"color: white; background-color: {self.colors[self.component.component_type].name()}; padding: 2px;")
        
        self.name_label = QLabel(self.component.name)
        self.name_label.setStyleSheet("font-weight: bold;")
        
        header_layout.addWidget(self.type_label)
        header_layout.addWidget(self.name_label, 1)
        
        layout.addLayout(header_layout)
        
        # Component specific content
        content_layout = QGridLayout()
        
        if self.component.component_type == 'indicator':
            content_layout.addWidget(QLabel("Type:"), 0, 0)
            content_layout.addWidget(QLabel(self.component.indicator_type.upper()), 0, 1)
            
            content_layout.addWidget(QLabel("Period:"), 1, 0)
            content_layout.addWidget(QLabel(str(self.component.params.get('period', 14))), 1, 1)
            
            content_layout.addWidget(QLabel("Source:"), 2, 0)
            content_layout.addWidget(QLabel(self.component.params.get('source', 'close')), 2, 1)
            
        elif self.component.component_type == 'filter':
            content_layout.addWidget(QLabel("Type:"), 0, 0)
            content_layout.addWidget(QLabel(self.component.filter_type.capitalize()), 0, 1)
            
            if self.component.filter_type == 'threshold':
                content_layout.addWidget(QLabel("Comparison:"), 1, 0)
                content_layout.addWidget(QLabel(self.component.params.get('comparison', 'above')), 1, 1)
                
                content_layout.addWidget(QLabel("Threshold:"), 2, 0)
                content_layout.addWidget(QLabel(str(self.component.params.get('threshold', 70))), 2, 1)
                
        elif self.component.component_type == 'signal':
            content_layout.addWidget(QLabel("Type:"), 0, 0)
            content_layout.addWidget(QLabel(self.component.signal_type.capitalize()), 0, 1)
            
            content_layout.addWidget(QLabel("Action:"), 1, 0)
            content_layout.addWidget(QLabel(self.component.params.get('action', 'buy')), 1, 1)
            
        layout.addLayout(content_layout)
        
        # Connection points
        conn_layout = QHBoxLayout()
        
        if self.component.component_type in ['indicator', 'filter']:
            self.output_btn = QPushButton("→")
            self.output_btn.setFixedSize(20, 20)
            self.output_btn.setToolTip("Output")
            conn_layout.addStretch()
            conn_layout.addWidget(self.output_btn)
            
        if self.component.component_type in ['filter', 'signal']:
            self.input_btn = QPushButton("←")
            self.input_btn.setFixedSize(20, 20)
            self.input_btn.setToolTip("Input")
            conn_layout.insertWidget(0, self.input_btn)
            conn_layout.insertStretch(1)
            
        layout.addLayout(conn_layout)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.offset = event.pos()
        self.selected.emit(self)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if event.buttons() & Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.component.component_id)
            drag.setMimeData(mime_data)
            
            # Create transparent pixmap for drag
            pixmap = QPixmap(self.size())
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setOpacity(0.7)
            self.render(painter)
            painter.end()
            
            drag.setPixmap(pixmap)
            drag.setHotSpot(self.offset)
            
            drag.exec_(Qt.MoveAction)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        super().mouseReleaseEvent(event)
        
    def set_position(self, pos):
        """Set widget position"""
        self.move(pos)
        self.moved.emit(pos)

class StrategyCanvas(QWidget):
    """Canvas for visually building strategies"""
    
    component_selected = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Strategy components
        self.components = {}
        self.connections = []
        
        # Component positions
        self.positions = {}
        
        # Selection tracking
        self.selected_component = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up canvas UI"""
        # Create scrollable canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.canvas = QWidget()
        self.canvas.setMinimumSize(2000, 1500)
        self.canvas_layout = None  # We'll use absolute positioning
        
        self.scroll_area.setWidget(self.canvas)
        self.layout.addWidget(self.scroll_area)
        
    def add_component(self, component, position=None):
        """Add component to canvas"""
        if component.component_id in self.components:
            logging.warning(f"Component {component.component_id} already exists")
            return
            
        # Create widget for component
        widget = ComponentWidget(component)
        widget.selected.connect(self.on_component_selected)
        
        # Add to tracking
        self.components[component.component_id] = widget
        
        # Set position if provided
        if position:
            self.positions[component.component_id] = position
            widget.set_position(position)
        else:
            # Default position
            self.positions[component.component_id] = QPoint(100, 100)
            widget.set_position(QPoint(100, 100))
            
        # Add to canvas
        widget.setParent(self.canvas)
        widget.show()
        
    def remove_component(self, component_id):
        """Remove component from canvas"""
        if component_id in self.components:
            # Remove widget
            widget = self.components[component_id]
            widget.deleteLater()
            
            # Remove from tracking
            del self.components[component_id]
            if component_id in self.positions:
                del self.positions[component_id]
                
            # Remove connections
            self.connections = [conn for conn in self.connections 
                               if conn[0] != component_id and conn[1] != component_id]
            
    def clear_canvas(self):
        """Clear all components from canvas"""
        for component_id in list(self.components.keys()):
            self.remove_component(component_id)
            
        self.connections = []
        self.selected_component = None
        
    def add_connection(self, source_id, target_id):
        """Add connection between components"""
        # Check if components exist
        if source_id not in self.components or target_id not in self.components:
            logging.error(f"Cannot connect: component not found")
            return False
            
        # Check if connection already exists
        if (source_id, target_id) in self.connections:
            logging.warning(f"Connection already exists")
            return False
            
        # Add connection
        self.connections.append((source_id, target_id))
        
        # Update component inputs/outputs
        source_component = self.components[source_id].component
        target_component = self.components[target_id].component
        
        if target_id not in source_component.outputs:
            source_component.outputs.append(target_id)
            
        if source_id not in target_component.inputs:
            target_component.inputs.append(source_id)
            
        # Trigger repaint
        self.update()
        
        return True
        
    def remove_connection(self, source_id, target_id):
        """Remove connection between components"""
        if (source_id, target_id) in self.connections:
            self.connections.remove((source_id, target_id))
            
            # Update component inputs/outputs
            if source_id in self.components:
                source_component = self.components[source_id].component
                if target_id in source_component.outputs:
                    source_component.outputs.remove(target_id)
                    
            if target_id in self.components:
                target_component = self.components[target_id].component
                if source_id in target_component.inputs:
                    target_component.inputs.remove(source_id)
                    
            # Trigger repaint
            self.update()
            
            return True
            
        return False
        
    def on_component_selected(self, component_widget):
        """Handle component selection"""
        self.selected_component = component_widget
        self.component_selected.emit(component_widget.component)
        
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        mime_data = event.mimeData()
        if mime_data.hasText():
            component_id = mime_data.text()
            if component_id in self.components:
                event.acceptProposedAction()
                
    def dragMoveEvent(self, event):
        """Handle drag move events"""
        event.acceptProposedAction()
        
    def dropEvent(self, event):
        """Handle drop events"""
        mime_data = event.mimeData()
        if mime_data.hasText():
            component_id = mime_data.text()
            if component_id in self.components:
                # Update component position
                widget = self.components[component_id]
                pos = event.pos() - widget.offset
                
                # Keep within canvas bounds
                pos.setX(max(0, min(pos.x(), self.canvas.width() - widget.width())))
                pos.setY(max(0, min(pos.y(), self.canvas.height() - widget.height())))
                
                widget.set_position(pos)
                self.positions[component_id] = pos
                
                event.acceptProposedAction()
                
    def paintEvent(self, event):
        """Handle paint events to draw connections"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections between components
        painter.setPen(Qt.black)
        
        for source_id, target_id in self.connections:
            if source_id in self.components and target_id in self.components:
                source_widget = self.components[source_id]
                target_widget = self.components[target_id]
                
                # Calculate connection points
                source_point = source_widget.geometry().topRight()
                target_point = target_widget.geometry().topLeft()
                
                # Draw arrow line
                painter.drawLine(source_point, target_point)
                
                # Draw arrowhead
                angle = atan2(target_point.y() - source_point.y(), 
                             target_point.x() - source_point.x())
                arrow_size = 10
                
                arrow_p1 = QPoint(
                    target_point.x() - arrow_size * cos(angle - pi/6),
                    target_point.y() - arrow_size * sin(angle - pi/6)
                )
                
                arrow_p2 = QPoint(
                    target_point.x() - arrow_size * cos(angle + pi/6),
                    target_point.y() - arrow_size * sin(angle + pi/6)
                )
                
                arrow_points = [target_point, arrow_p1, arrow_p2]
                painter.setBrush(Qt.black)
                painter.drawPolygon(arrow_points)

class ComponentProperties(QWidget):
    """Widget for editing component properties"""
    
    properties_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.component = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up properties UI"""
        # Title
        self.title_label = QLabel("Component Properties")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title_label)
        
        # Properties container
        self.properties_container = QWidget()
        self.properties_layout = QVBoxLayout(self.properties_container)
        
        self.layout.addWidget(self.properties_container)
        
        # Initially hide properties
        self.properties_container.hide()
        
    def set_component(self, component):
        """Set component for editing properties"""
        self.component = component
        self.update_properties_ui()
        
    def update_properties_ui(self):
        """Update properties UI for current component"""
        # Clear existing widgets
        for i in reversed(range(self.properties_layout.count())):
            widget = self.properties_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
                
        if not self.component:
            self.properties_container.hide()
            return
            
        # Show properties container
        self.properties_container.show()
        
        # Create widgets for component properties
        # Component name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self.component.name)
        self.name_edit.textChanged.connect(self.on_name_changed)
        name_layout.addWidget(self.name_edit)
        self.properties_layout.addLayout(name_layout)
        
        # Component type-specific properties
        if self.component.component_type == 'indicator':
            # Indicator type
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Indicator:"))
            self.indicator_type = QComboBox()
            self.indicator_type.addItems(['sma', 'ema', 'rsi', 'macd', 'bbands'])
            self.indicator_type.setCurrentText(self.component.indicator_type)
            self.indicator_type.currentTextChanged.connect(self.on_indicator_type_changed)
            type_layout.addWidget(self.indicator_type)
            self.properties_layout.addLayout(type_layout)
            
            # Period
            period_layout = QHBoxLayout()
            period_layout.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 200)
            self.period_spin.setValue(self.component.params.get('period', 14))
            self.period_spin.valueChanged.connect(self.on_period_changed)
            period_layout.addWidget(self.period_spin)
            self.properties_layout.addLayout(period_layout)
            
            # Source
            source_layout = QHBoxLayout()
            source_layout.addWidget(QLabel("Source:"))
            self.source_combo = QComboBox()
            self.source_combo.addItems(['open', 'high', 'low', 'close', 'volume'])
            self.source_combo.setCurrentText(self.component.params.get('source', 'close'))
            self.source_combo.currentTextChanged.connect(self.on_source_changed)
            source_layout.addWidget(self.source_combo)
            self.properties_layout.addLayout(source_layout)
            
        elif self.component.component_type == 'filter':
            # Filter type
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Filter:"))
            self.filter_type = QComboBox()
            self.filter_type.addItems(['threshold', 'crossover'])
            self.filter_type.setCurrentText(self.component.filter_type)
            self.filter_type.currentTextChanged.connect(self.on_filter_type_changed)
            type_layout.addWidget(self.filter_type)
            self.properties_layout.addLayout(type_layout)
            
            if self.component.filter_type == 'threshold':
                # Comparison
                comparison_layout = QHBoxLayout()
                comparison_layout.addWidget(QLabel("Comparison:"))
                self.comparison_combo = QComboBox()
                self.comparison_combo.addItems(['above', 'below', 'equal', 'between'])
                self.comparison_combo.setCurrentText(self.component.params.get('comparison', 'above'))
                self.comparison_combo.currentTextChanged.connect(self.on_comparison_changed)
                comparison_layout.addWidget(self.comparison_combo)
                self.properties_layout.addLayout(comparison_layout)
                
                # Threshold
                threshold_layout = QHBoxLayout()
                threshold_layout.addWidget(QLabel("Threshold:"))
                self.threshold_spin = QDoubleSpinBox()
                self.threshold_spin.setRange(-1000, 1000)
                self.threshold_spin.setValue(self.component.params.get('threshold', 70))
                self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
                threshold_layout.addWidget(self.threshold_spin)
                self.properties_layout.addLayout(threshold_layout)
                
                # Lower/Upper thresholds for 'between' comparison
                if self.component.params.get('comparison', 'above') == 'between':
                    lower_layout = QHBoxLayout()
                    lower_layout.addWidget(QLabel("Lower Threshold:"))
                    self.lower_spin = QDoubleSpinBox()
                    self.lower_spin.setRange(-1000, 1000)
                    self.lower_spin.setValue(self.component.params.get('lower_threshold', 30))
                    self.lower_spin.valueChanged.connect(self.on_lower_threshold_changed)
                    lower_layout.addWidget(self.lower_spin)
                    self.properties_layout.addLayout(lower_layout)
                    
                    upper_layout = QHBoxLayout()
                    upper_layout.addWidget(QLabel("Upper Threshold:"))
                    self.upper_spin = QDoubleSpinBox()
                    self.upper_spin.setRange(-1000, 1000)
                    self.upper_spin.setValue(self.component.params.get('upper_threshold', 70))
                    self.upper_spin.valueChanged.connect(self.on_upper_threshold_changed)
                    upper_layout.addWidget(self.upper_spin)
                    self.properties_layout.addLayout(upper_layout)
                    
        elif self.component.component_type == 'signal':
            # Signal type
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Signal:"))
            self.signal_type = QComboBox()
            self.signal_type.addItems(['simple', 'conditional', 'multi_condition'])
            self.signal_type.setCurrentText(self.component.signal_type)
            self.signal_type.currentTextChanged.connect(self.on_signal_type_changed)
            type_layout.addWidget(self.signal_type)
            self.properties_layout.addLayout(type_layout)
            
            # Action
            action_layout = QHBoxLayout()
            action_layout.addWidget(QLabel("Action:"))
            self.action_combo = QComboBox()
            self.action_combo.addItems(['buy', 'sell', 'hold'])
            self.action_combo.setCurrentText(self.component.params.get('action', 'buy'))
            self.action_combo.currentTextChanged.connect(self.on_action_changed)
            action_layout.addWidget(self.action_combo)
            self.properties_layout.addLayout(action_layout)
            
        # Add apply button
        apply_btn = QPushButton("Apply Changes")
        apply_btn.clicked.connect(self.apply_changes)
        self.properties_layout.addWidget(apply_btn)
        
        # Add stretch to bottom
        self.properties_layout.addStretch()
        
    def apply_changes(self):
        """Apply all changes to component"""
        if not self.component:
            return
            
        # Emit signal that properties changed
        self.properties_changed.emit(self.component)
        
    def on_name_changed(self, name):
        """Handle name edit"""
        if self.component:
            self.component.name = name
            
    def on_indicator_type_changed(self, indicator_type):
        """Handle indicator type change"""
        if self.component and self.component.component_type == 'indicator':
            self.component.indicator_type = indicator_type
            self.update_properties_ui()
            
    def on_period_changed(self, period):
        """Handle period change"""
        if self.component and self.component.component_type == 'indicator':
            self.component.params['period'] = period
            
    def on_source_changed(self, source):
        """Handle source change"""
        if self.component and self.component.component_type == 'indicator':
            self.component.params['source'] = source
            
    def on_filter_type_changed(self, filter_type):
        """Handle filter type change"""
        if self.component and self.component.component_type == 'filter':
            self.component.filter_type = filter_type
            self.update_properties_ui()
            
    def on_comparison_changed(self, comparison):
        """Handle comparison change"""
        if self.component and self.component.component_type == 'filter':
            self.component.params['comparison'] = comparison
            self.update_properties_ui()
            
    def on_threshold_changed(self, threshold):
        """Handle threshold change"""
        if self.component and self.component.component_type == 'filter':
            self.component.params['threshold'] = threshold
            
    def on_lower_threshold_changed(self, threshold):
        """Handle lower threshold change"""
        if self.component and self.component.component_type == 'filter':
            self.component.params['lower_threshold'] = threshold
            
    def on_upper_threshold_changed(self, threshold):
        """Handle upper threshold change"""
        if self.component and self.component.component_type == 'filter':
            self.component.params['upper_threshold'] = threshold
            
    def on_signal_type_changed(self, signal_type):
        """Handle signal type change"""
        if self.component and self.component.component_type == 'signal':
            self.component.signal_type = signal_type
            self.update_properties_ui()
            
    def on_action_changed(self, action):
        """Handle action change"""
        if self.component and self.component.component_type == 'signal':
            self.component.params['action'] = action

class ComponentPalette(QWidget):
    """Widget for selecting components to add to strategy"""
    
    component_added = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up palette UI"""
        # Title
        self.title_label = QLabel("Component Palette")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title_label)
        
        # Components organized by type
        self.indicators_group = QGroupBox("Indicators")
        indicators_layout = QVBoxLayout(self.indicators_group)
        
        # Add indicator buttons
        for indicator_type in ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands']:
            btn = QPushButton(indicator_type)
            btn.clicked.connect(lambda checked, t=indicator_type.lower().replace(' ', '_'): 
                               self.add_indicator(t))
            indicators_layout.addWidget(btn)
            
        self.layout.addWidget(self.indicators_group)
        
        # Filters group
        self.filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout(self.filters_group)
        
        # Add filter buttons
        for filter_type in ['Threshold', 'Crossover']:
            btn = QPushButton(filter_type)
            btn.clicked.connect(lambda checked, t=filter_type.lower(): 
                               self.add_filter(t))
            filters_layout.addWidget(btn)
            
        self.layout.addWidget(self.filters_group)
        
        # Signals group
        self.signals_group = QGroupBox("Signals")
        signals_layout = QVBoxLayout(self.signals_group)
        
        # Add signal buttons
        for signal_type in ['Simple', 'Conditional', 'Multi Condition']:
            btn = QPushButton(signal_type)
            btn.clicked.connect(lambda checked, t=signal_type.lower().replace(' ', '_'): 
                               self.add_signal(t))
            signals_layout.addWidget(btn)
            
        self.layout.addWidget(self.signals_group)
        
        # Add stretch to bottom
        self.layout.addStretch()
        
    def add_indicator(self, indicator_type):
        """Add indicator component"""
        component = IndicatorComponent(indicator_type=indicator_type)
        self.component_added.emit(component)
        
    def add_filter(self, filter_type):
        """Add filter component"""
        component = FilterComponent(filter_type=filter_type)
        self.component_added.emit(component)
        
    def add_signal(self, signal_type):
        """Add signal component"""
        component = SignalComponent(signal_type=signal_type)
        self.component_added.emit(component)

class StrategyBuilderWindow(QWidget):
    """Main window for strategy builder"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Strategy Builder")
        self.resize(1200, 800)
        
        # Current strategy
        self.current_strategy = None
        self.strategy_name = "New Strategy"
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up main UI"""
        self.layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        # Strategy name
        toolbar_layout.addWidget(QLabel("Strategy:"))
        self.strategy_name_edit = QLineEdit(self.strategy_name)
        toolbar_layout.addWidget(self.strategy_name_edit)
        
        # Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_strategy)
        toolbar_layout.addWidget(self.save_btn)
        
        # Load button
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_strategy)
        toolbar_layout.addWidget(self.load_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_strategy)
        toolbar_layout.addWidget(self.clear_btn)
        
        # Test button
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.test_strategy)
        toolbar_layout.addWidget(self.test_btn)
        
        self.layout.addLayout(toolbar_layout)
        
        # Main content - split view
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side - component palette
        self.component_palette = ComponentPalette()
        self.component_palette.component_added.connect(self.add_component)
        self.splitter.addWidget(self.component_palette)
        
        # Middle - canvas
        self.strategy_canvas = StrategyCanvas()
        self.strategy_canvas.component_selected.connect(self.component_selected)
        self.splitter.addWidget(self.strategy_canvas)
        
        # Right side - component properties
        self.component_properties = ComponentProperties()
        self.component_properties.properties_changed.connect(self.component_properties_changed)
        self.splitter.addWidget(self.component_properties)
        
        # Set initial splitter sizes
        self.splitter.setSizes([200, 600, 300])
        
        self.layout.addWidget(self.splitter)
        
    def add_component(self, component):
        """Add component to canvas"""
        # Find a suitable position
        pos = QPoint(100, 100)
        self.strategy_canvas.add_component(component, pos)
        
    def component_selected(self, component):
        """Handle component selection"""
        self.component_properties.set_component(component)
        
    def component_properties_changed(self, component):
        """Handle component property changes"""
        # Update component widget
        if component.component_id in self.strategy_canvas.components:
            widget = self.strategy_canvas.components[component.component_id]
            
            # Remove old widget and add new one
            pos = widget.pos()
            self.strategy_canvas.remove_component(component.component_id)
            self.strategy_canvas.add_component(component, pos)
        
    def save_strategy(self):
        """Save current strategy"""
        try:
            # Get strategy name
            strategy_name = self.strategy_name_edit.text()
            if not strategy_name:
                QMessageBox.warning(self, "Save Strategy", "Please enter a strategy name")
                return
                
            # Collect strategy data
            strategy_data = {
                'name': strategy_name,
                'components': [],
                'connections': self.strategy_canvas.connections,
                'positions': {}
            }
            
            # Collect component data
            for component_id, widget in self.strategy_canvas.components.items():
                component_data = widget.component.to_dict()
                strategy_data['components'].append(component_data)
                
                # Save position
                pos = widget.pos()
                strategy_data['positions'][component_id] = {'x': pos.x(), 'y': pos.y()}
                
            # Save to file
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Strategy", "", "Strategy Files (*.json)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(strategy_data, f, indent=4)
                    
                self.current_strategy = filename
                QMessageBox.information(self, "Save Strategy", f"Strategy saved to {filename}")
                
        except Exception as e:
            logging.error(f"Error saving strategy: {e}")
            QMessageBox.critical(self, "Save Error", f"Error saving strategy: {str(e)}")
            
    def load_strategy(self):
        """Load strategy from file"""
        try:
            # Ask for file
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Strategy", "", "Strategy Files (*.json)"
            )
            
            if not filename:
                return
                
            # Load strategy data
            with open(filename, 'r') as f:
                strategy_data = json.load(f)
                
            # Clear current strategy
            self.strategy_canvas.clear_canvas()
            
            # Set strategy name
            strategy_name = strategy_data.get('name', "Loaded Strategy")
            self.strategy_name_edit.setText(strategy_name)
            
            # Add components
            for component_data in strategy_data.get('components', []):
                component_id = component_data.get('component_id')
                component_type = component_data.get('component_type')
                
                # Create component based on type
                if component_type == 'indicator':
                    component = IndicatorComponent.from_dict(component_data)
                elif component_type == 'filter':
                    component = FilterComponent.from_dict(component_data)
                elif component_type == 'signal':
                    component = SignalComponent.from_dict(component_data)
                else:
                    logging.warning(f"Unknown component type: {component_type}")
                    continue
                    
                # Get position
                position = None
                if component_id in strategy_data.get('positions', {}):
                    pos_data = strategy_data['positions'][component_id]
                    position = QPoint(pos_data.get('x', 100), pos_data.get('y', 100))
                    
                # Add to canvas
                self.strategy_canvas.add_component(component, position)
                
            # Add connections
            for source_id, target_id in strategy_data.get('connections', []):
                self.strategy_canvas.add_connection(source_id, target_id)
                
            self.current_strategy = filename
            QMessageBox.information(self, "Load Strategy", f"Strategy loaded from {filename}")
            
        except Exception as e:
            logging.error(f"Error loading strategy: {e}")
            QMessageBox.critical(self, "Load Error", f"Error loading strategy: {str(e)}")
            
    def clear_strategy(self):
        """Clear current strategy"""
        reply = QMessageBox.question(
            self, "Clear Strategy", 
            "Are you sure you want to clear the current strategy?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.strategy_canvas.clear_canvas()
            self.current_strategy = None
            self.strategy_name_edit.setText("New Strategy")
            
    def test_strategy(self):
        """Test current strategy with sample data"""
        try:
            # Check if we have a valid strategy
            if not self.strategy_canvas.components:
                QMessageBox.warning(self, "Test Strategy", "No components to test")
                return
                
            # Create topological sort of components
            components = []
            
            # Start with components that have no inputs (sources)
            sources = []
            for component_id, widget in self.strategy_canvas.components.items():
                if not widget.component.inputs:
                    sources.append(component_id)
                    
            # Process components in order
            processed = set()
            while sources:
                component_id = sources.pop(0)
                if component_id in processed:
                    continue
                    
                # Add to processed list
                processed.add(component_id)
                components.append(self.strategy_canvas.components[component_id].component)
                
                # Add outputs to sources
                widget = self.strategy_canvas.components[component_id]
                for target_id in widget.component.outputs:
                    if target_id in self.strategy_canvas.components:
                        target_widget = self.strategy_canvas.components[target_id]
                        
                        # Check if all inputs are processed
                        all_inputs_processed = True
                        for input_id in target_widget.component.inputs:
                            if input_id not in processed:
                                all_inputs_processed = False
                                break
                                
                        if all_inputs_processed:
                            sources.append(target_id)
                            
            # If components list doesn't include all components, there might be a cycle
            if len(components) != len(self.strategy_canvas.components):
                QMessageBox.warning(self, "Test Strategy", "Strategy contains cycles")
                return
                
            # Show components in order
            component_list = "\n".join([f"{i+1}. {c.name}" for i, c in enumerate(components)])
            QMessageBox.information(
                self, "Test Strategy", 
                f"Strategy components in execution order:\n\n{component_list}"
            )
            
        except Exception as e:
            logging.error(f"Error testing strategy: {e}")
            QMessageBox.critical(self, "Test Error", f"Error testing strategy: {str(e)}")

# Add this function to your OptimizedTradingGUI class
def open_strategy_builder(self):
    """Open the strategy builder window"""
    self.strategy_builder = StrategyBuilderWindow(self)
    self.strategy_builder.show()
