# status_widget.py

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, 
                           QVBoxLayout, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, pyqtProperty
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QPalette

class StatusIndicator(QFrame):
    """
    A colored circular indicator that shows status through color changes.
    Can optionally display a status value.
    """
    
    # Status color constants
    STATUS_COLORS = {
        "default": QColor("#888888"),  # Gray (default/unknown)
        "good": QColor("#4CAF50"),     # Green
        "warning": QColor("#FFC107"),  # Yellow/Amber
        "error": QColor("#F44336"),    # Red
        "info": QColor("#2196F3"),     # Blue
        "inactive": QColor("#9E9E9E")  # Dark Gray
    }
    
    def __init__(self, parent=None, size=16, status="default", show_value=False):
        """
        Initialize the status indicator.
        
        Args:
            parent: Parent widget
            size: Size of the indicator in pixels
            status: Initial status ("default", "good", "warning", "error", "info")
            show_value: Whether to show a value label next to the indicator
        """
        super().__init__(parent)
        
        # Initialize instance variables
        self._size = size
        self._status = status
        self._color = self.STATUS_COLORS.get(status, self.STATUS_COLORS["default"])
        self._custom_color = None
        self._show_value = show_value
        self.value_widget = None  # Initialize to avoid AttributeError
        
        # Configure frame
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(self._size, self._size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Set up layout
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        # If we need to show a value, create a layout with the indicator and value
        if self._show_value:
            # Create a container widget with horizontal layout
            container = QWidget(self.parent())
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)
            
            # Add the indicator (self) to the layout
            layout.addWidget(self)
            
            # Create and add the value label
            self.value_widget = QLabel("--")
            self.value_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(self.value_widget)
            
            # Ensure the container is laid out properly
            container.setLayout(layout)
        
    def paintEvent(self, event):
        """Custom paint event to draw the circular indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate dimensions
        rect = self.rect()
        diameter = min(rect.width(), rect.height()) - 2
        x = (rect.width() - diameter) / 2
        y = (rect.height() - diameter) / 2
        
        # Draw border
        border_color = QColor(Qt.black)
        border_color.setAlpha(40)  # Semi-transparent border
        painter.setPen(QPen(border_color, 1))
        
        # Draw filled circle with current status color
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(int(x), int(y), diameter, diameter)
        
    def setStatus(self, status):
        """Set the status and update the color"""
        self._status = status
        self._color = self.STATUS_COLORS.get(status, self.STATUS_COLORS["default"])
        self.update()
        
    def status(self):
        """Get the current status"""
        return self._status
        
    def setCustomColor(self, color):
        """Set a custom color for the indicator"""
        if isinstance(color, QColor):
            self._custom_color = color
            self._color = color
        else:
            self._custom_color = QColor(color)
            self._color = self._custom_color
        self.update()
        
    def customColor(self):
        """Get the custom color if set"""
        return self._custom_color
        
    def setSize(self, size):
        """Set the size of the indicator"""
        self._size = size
        self.setFixedSize(self._size, self._size)
        self.update()
        
    def size(self):
        """Get the size of the indicator"""
        return self._size
        
    def setValue(self, value):
        """Set the displayed value"""
        # Check if value_widget exists before attempting to use it
        if hasattr(self, 'value_widget') and self.value_widget is not None:
            self.value_widget.setText(str(value))
            
    def updateColor(self, value, thresholds=None):
        """
        Update indicator color based on value and thresholds.
        
        Args:
            value: The value to evaluate
            thresholds: Dict with keys 'good', 'warning', 'error' and corresponding threshold values
        """
        # First, make sure the value_widget attribute exists
        if not hasattr(self, 'value_widget'):
            self.value_widget = None
            
        # Update value if value widget exists
        if self.value_widget is not None:
            self.value_widget.setText(str(value))
            
        # Use default thresholds if none provided
        if thresholds is None:
            thresholds = {
                'good': 80,
                'warning': 50,
                'error': 20
            }
            
        # Determine status based on thresholds
        try:
            # Convert value to float for comparison
            float_value = float(value)
            
            if float_value >= thresholds.get('good', 80):
                self.setStatus('good')
            elif float_value >= thresholds.get('warning', 50):
                self.setStatus('warning')
            elif float_value >= thresholds.get('error', 20):
                self.setStatus('error')
            else:
                self.setStatus('error')
        except (ValueError, TypeError):
            # If value can't be converted to float, use default status
            self.setStatus('default')

    # Define Qt properties
    status_prop = pyqtProperty(str, status, setStatus)
    size_prop = pyqtProperty(int, size, setSize)
    color_prop = pyqtProperty(QColor, customColor, setCustomColor)

class StatusWidget(QWidget):
    """
    A status display widget that shows a label and a status indicator.
    Optionally shows a value.
    """
    
    def __init__(self, label="Status", parent=None, show_value=False, size=16, status="default"):
        """
        Initialize the status widget.
        
        Args:
            label: Label text
            parent: Parent widget
            show_value: Whether to show a value
            size: Size of the indicator
            status: Initial status
        """
        super().__init__(parent)
        
        # Initialize instance variables
        self.label_text = label
        self.show_value = show_value
        
        # Create layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(5)
        
        # Create and add label
        self.label = QLabel(label)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.label)
        
        # Create and add status indicator
        self.indicator = StatusIndicator(self, size=size, status=status, show_value=show_value)
        self.layout.addWidget(self.indicator)
        
        # If show_value, reference the inner value_widget
        if show_value and hasattr(self.indicator, 'value_widget'):
            self.value_widget = self.indicator.value_widget
        else:
            self.value_widget = None
        
    def setStatus(self, status):
        """Set the status of the indicator"""
        self.indicator.setStatus(status)
        
    def setValue(self, value):
        """Set the displayed value"""
        self.indicator.setValue(value)
        
    def updateColor(self, value, thresholds=None):
        """Update indicator color based on value and thresholds"""
        self.indicator.updateColor(value, thresholds)
        
    def setCustomColor(self, color):
        """Set a custom color for the indicator"""
        self.indicator.setCustomColor(color)

class SystemStatusPanel(QFrame):
    """
    A panel showing multiple system status indicators.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configure frame
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        
        # Title
        self.title = QLabel("System Status")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title)
        
        # Status indicators
        self.status_widgets = {}
        
        # Add common system status indicators
        self._add_status_indicators()
        
        # Add stretch to bottom
        self.layout.addStretch()
        
    def _add_status_indicators(self):
        """Add the default set of status indicators"""
        # Database connection
        self.add_status_indicator("Database", "default", True)
        
        # API connection
        self.add_status_indicator("API Connection", "default", True)
        
        # Strategies
        self.add_status_indicator("Strategies", "default", True)
        
        # Data Feed
        self.add_status_indicator("Data Feed", "default", True)
        
        # Risk Management
        self.add_status_indicator("Risk System", "default", True)
        
        # Machine Learning
        self.add_status_indicator("ML Models", "default", True)
        
    def add_status_indicator(self, label, initial_status="default", show_value=False):
        """
        Add a new status indicator to the panel.
        
        Args:
            label: Label for the indicator
            initial_status: Initial status
            show_value: Whether to show a value
        """
        status_widget = StatusWidget(label, self, show_value, 16, initial_status)
        self.layout.addWidget(status_widget)
        self.status_widgets[label] = status_widget
        return status_widget
        
    def update_status(self, label, status, value=None):
        """
        Update a status indicator.
        
        Args:
            label: Label of the indicator to update
            status: New status
            value: New value (if applicable)
        """
        if label in self.status_widgets:
            widget = self.status_widgets[label]
            widget.setStatus(status)
            
            if value is not None:
                widget.setValue(value)
                
    def update_all_statuses(self, status_dict):
        """
        Update multiple status indicators at once.
        
        Args:
            status_dict: Dict mapping labels to (status, value) tuples
        """
        for label, (status, value) in status_dict.items():
            self.update_status(label, status, value)
