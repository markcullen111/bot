# gui/widgets/alert_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFrame, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QEvent
from PyQt5.QtGui import QColor, QIcon, QFont

class AlertItem(QFrame):
    """Individual alert notification item."""
    
    dismissed = pyqtSignal(str)  # Signal emitted when alert is dismissed
    action_triggered = pyqtSignal(str, str)  # Signal for alert action (alert_id, action)
    
    def __init__(self, alert_id, message, alert_type="info", timeout=0, actions=None, parent=None):
        """
        Initialize alert item.
        
        Args:
            alert_id (str): Unique identifier for this alert
            message (str): Alert message to display
            alert_type (str): Type of alert ('info', 'warning', 'error', 'success')
            timeout (int): Auto-dismiss timeout in ms (0 = no auto-dismiss)
            actions (list): List of action button texts
            parent (QWidget): Parent widget
        """
        super().__init__(parent)
        self.alert_id = alert_id
        self.message = message
        self.alert_type = alert_type
        self.timeout = timeout
        self.actions = actions or []
        
        # Setup UI
        self._init_ui()
        
        # Setup auto-dismiss timer if timeout > 0
        if self.timeout > 0:
            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self._auto_dismiss)
            self.timer.start(self.timeout)
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        
        # Set background color based on alert type
        colors = {
            "info": "#E3F2FD",     # Light blue
            "warning": "#FFF8E1",  # Light amber
            "error": "#FFEBEE",    # Light red
            "success": "#E8F5E9"   # Light green
        }
        self.setStyleSheet(f"background-color: {colors.get(self.alert_type, colors['info'])}; border-radius: 4px;")
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Add icon based on alert type
        icon_label = QLabel()
        icon_label.setFixedSize(24, 24)
        icon_names = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        icon_label.setText(icon_names.get(self.alert_type, "ℹ️"))
        layout.addWidget(icon_label)
        
        # Add message
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(message_label, 1)
        
        # Add action buttons if any
        if self.actions:
            for action in self.actions:
                btn = QPushButton(action)
                btn.setFlat(True)
                btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(self.alert_id, a))
                layout.addWidget(btn)
        
        # Add dismiss button
        dismiss_btn = QPushButton("×")
        dismiss_btn.setFlat(True)
        dismiss_btn.setFixedSize(20, 20)
        dismiss_btn.setToolTip("Dismiss")
        dismiss_btn.clicked.connect(self._dismiss)
        layout.addWidget(dismiss_btn)
    
    def _dismiss(self):
        """Dismiss this alert."""
        self.dismissed.emit(self.alert_id)
    
    def _auto_dismiss(self):
        """Auto-dismiss after timeout."""
        self._dismiss()


class AlertWidget(QWidget):
    """Widget for displaying alert notifications."""
    
    alert_dismissed = pyqtSignal(str)  # Signal when alert is dismissed
    alert_action = pyqtSignal(str, str)  # Signal when alert action is triggered
    
    def __init__(self, max_alerts=5, parent=None):
        """
        Initialize alert widget.
        
        Args:
            max_alerts (int): Maximum number of alerts to display
            parent (QWidget): Parent widget
        """
        super().__init__(parent)
        self.max_alerts = max_alerts
        self.alerts = {}  # Map of alert_id to AlertItem
        self.alert_queue = []  # Queue of alerts waiting to be displayed
        
        # Setup UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        
        # Scroll area for alerts
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget for alerts
        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_container)
        self.alert_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_layout.setSpacing(4)
        self.alert_layout.addStretch()  # Push alerts to the top
        
        self.scroll_area.setWidget(self.alert_container)
        self.layout.addWidget(self.scroll_area)
    
    def add_alert(self, message, alert_type="info", timeout=5000, actions=None, alert_id=None):
        """
        Add a new alert notification.
        
        Args:
            message (str): Alert message
            alert_type (str): Type of alert ('info', 'warning', 'error', 'success')
            timeout (int): Auto-dismiss timeout in ms (0 = no auto-dismiss)
            actions (list): List of action button texts
            alert_id (str): Optional alert ID, generated if not provided
            
        Returns:
            str: Alert ID
        """
        # Generate alert ID if not provided
        if alert_id is None:
            import uuid
            alert_id = str(uuid.uuid4())
        
        # Check if we can add more alerts
        if len(self.alerts) >= self.max_alerts:
            # Queue the alert
            self.alert_queue.append({
                'alert_id': alert_id,
                'message': message,
                'alert_type': alert_type,
                'timeout': timeout,
                'actions': actions
            })
            return alert_id
        
        # Create alert item
        alert_item = AlertItem(
            alert_id=alert_id,
            message=message,
            alert_type=alert_type,
            timeout=timeout,
            actions=actions,
            parent=self
        )
        
        # Connect signals
        alert_item.dismissed.connect(self._on_alert_dismissed)
        alert_item.action_triggered.connect(self._on_alert_action)
        
        # Add to layout (at position 0 to keep newest alerts at the top)
        self.alert_layout.insertWidget(0, alert_item)
        
        # Store in alerts map
        self.alerts[alert_id] = alert_item
        
        return alert_id
    
    def dismiss_alert(self, alert_id):
        """
        Dismiss an alert by ID.
        
        Args:
            alert_id (str): ID of alert to dismiss
            
        Returns:
            bool: True if alert was found and dismissed
        """
        if alert_id in self.alerts:
            alert_item = self.alerts[alert_id]
            self._remove_alert_item(alert_item)
            return True
        return False
    
    def dismiss_all_alerts(self):
        """Dismiss all active alerts."""
        for alert_id in list(self.alerts.keys()):
            self.dismiss_alert(alert_id)
    
    def _on_alert_dismissed(self, alert_id):
        """Handle alert dismissed signal."""
        self.dismiss_alert(alert_id)
        self.alert_dismissed.emit(alert_id)
        
        # Process queue if available
        self._process_queue()
    
    def _on_alert_action(self, alert_id, action):
        """Handle alert action triggered signal."""
        self.alert_action.emit(alert_id, action)
    
    def _remove_alert_item(self, alert_item):
        """Remove alert item from layout and map."""
        # Remove from layout
        self.alert_layout.removeWidget(alert_item)
        alert_item.setParent(None)
        alert_item.deleteLater()
        
        # Remove from map
        if alert_item.alert_id in self.alerts:
            del self.alerts[alert_item.alert_id]
    
    def _process_queue(self):
        """Process queued alerts if space available."""
        while self.alert_queue and len(self.alerts) < self.max_alerts:
            alert_data = self.alert_queue.pop(0)
            self.add_alert(
                message=alert_data['message'],
                alert_type=alert_data['alert_type'],
                timeout=alert_data['timeout'],
                actions=alert_data['actions'],
                alert_id=alert_data['alert_id']
            )
    
    def sizeHint(self):
        """Suggest a size for the widget."""
        return QSize(300, 200)


# Example usage:
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    # Create alert widget
    alert_widget = AlertWidget()
    
    # Connect signals
    alert_widget.alert_dismissed.connect(lambda alert_id: print(f"Alert {alert_id} dismissed"))
    alert_widget.alert_action.connect(lambda alert_id, action: print(f"Alert {alert_id}, action: {action}"))
    
    # Add some alerts
    alert_widget.add_alert("Info notification example", "info", 5000)
    alert_widget.add_alert("Warning! Unusual market volatility detected", "warning", 0, ["View Details"])
    alert_widget.add_alert("Error connecting to exchange API", "error", 0, ["Retry", "Settings"])
    alert_widget.add_alert("Trade successfully executed", "success", 3000)
    
    # Set up the main window
    window.setCentralWidget(alert_widget)
    window.setWindowTitle("Alert Widget Example")
    window.resize(400, 300)
    window.show()
    
    sys.exit(app.exec_())
