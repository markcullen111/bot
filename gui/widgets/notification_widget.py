# gui/widgets/notification_widget.py

import os
import time
import json
import queue
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QFrame, QSizePolicy, QGraphicsOpacityEffect,
    QApplication, QStyleOption, QStyle
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, 
    QRect, QSize, QPoint, QObject, pyqtProperty, QEvent
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QFont, QPalette, QPen,
    QFontMetrics, QCursor, QMouseEvent
)
from PyQt5.QtMultimedia import QSound

class NotificationType(Enum):
    """Enum defining notification types with associated properties."""
    SUCCESS = {
        "color": "#28a745",  # Green
        "icon": "success.png",
        "sound": "success.wav",
        "title_prefix": "Success",
        "auto_dismiss": True,
        "default_duration": 5000  # 5 seconds
    }
    INFO = {
        "color": "#17a2b8",  # Blue
        "icon": "info.png",
        "sound": "info.wav",
        "title_prefix": "Information",
        "auto_dismiss": True,
        "default_duration": 7000  # 7 seconds
    }
    WARNING = {
        "color": "#ffc107",  # Yellow
        "icon": "warning.png",
        "sound": "warning.wav",
        "title_prefix": "Warning",
        "auto_dismiss": False,
        "default_duration": 0  # Don't auto-dismiss
    }
    ERROR = {
        "color": "#dc3545",  # Red
        "icon": "error.png",
        "sound": "error.wav",
        "title_prefix": "Error",
        "auto_dismiss": False,
        "default_duration": 0  # Don't auto-dismiss
    }
    TRADE = {
        "color": "#6f42c1",  # Purple
        "icon": "trade.png",
        "sound": "trade.wav",
        "title_prefix": "Trade",
        "auto_dismiss": True,
        "default_duration": 10000  # 10 seconds
    }
    ALERT = {
        "color": "#fd7e14",  # Orange
        "icon": "alert.png",
        "sound": "alert.wav",
        "title_prefix": "Alert",
        "auto_dismiss": False,
        "default_duration": 0  # Don't auto-dismiss
    }
    SYSTEM = {
        "color": "#20c997",  # Teal
        "icon": "system.png",
        "sound": "system.wav",
        "title_prefix": "System",
        "auto_dismiss": True,
        "default_duration": 5000  # 5 seconds
    }

class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class NotificationAction:
    """Defines an action that can be taken on a notification."""
    
    def __init__(self, text: str, callback: Callable, data: Any = None):
        """
        Initialize notification action.
        
        Args:
            text: Button text to display
            callback: Function to call when action is triggered
            data: Optional data to pass to callback
        """
        self.text = text
        self.callback = callback
        self.data = data
        
    def trigger(self):
        """Trigger the action callback."""
        if self.callback:
            if self.data is not None:
                self.callback(self.data)
            else:
                self.callback()

class Notification:
    """
    Represents a single notification with comprehensive metadata.
    
    Attributes:
        id: Unique identifier
        type: Notification type (success, info, warning, etc.)
        title: Title text
        message: Main notification message
        created_at: Timestamp when notification was created
        duration: How long to display (ms), 0 = no auto-dismiss
        priority: Priority level for ordering
        actions: List of actions that can be taken
        sound_enabled: Whether to play sound
        data: Optional associated data
        dismissible: Whether user can manually dismiss
        progress: Optional progress value (0-100)
    """
    
    _counter = 0
    
    @classmethod
    def get_next_id(cls):
        """Generate a unique ID for each notification."""
        cls._counter += 1
        return f"notification_{cls._counter}"
    
    def __init__(
        self,
        type: NotificationType,
        message: str,
        title: str = None,
        duration: int = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        actions: List[NotificationAction] = None,
        sound_enabled: bool = True,
        data: Any = None,
        dismissible: bool = True,
        progress: int = None
    ):
        # Core properties
        self.id = self.get_next_id()
        self.type = type
        self.message = message
        self.title = title or type.value["title_prefix"]
        self.created_at = datetime.now()
        
        # Set duration based on type if not specified
        self.duration = duration if duration is not None else type.value["default_duration"]
        
        # Display and behavior properties
        self.priority = priority
        self.actions = actions or []
        self.sound_enabled = sound_enabled
        self.data = data
        self.dismissible = dismissible
        self.progress = progress  # 0-100, None if no progress
        
        # State tracking
        self.read = False
        self.dismissed = False
        self.expire_time = None
        
        # Set expire time if duration is specified
        if self.duration > 0:
            self.expire_time = time.time() + (self.duration / 1000)
            
    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expire_time is None:
            return False
        return time.time() > self.expire_time
        
    def mark_as_read(self):
        """Mark notification as read."""
        self.read = True
        
    def dismiss(self):
        """Dismiss the notification."""
        self.dismissed = True
        
    def __lt__(self, other):
        """Comparison for priority queue, higher priority comes first."""
        if not isinstance(other, Notification):
            return NotImplemented
        # First compare by priority (higher priority first)
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        # Then by creation time (newer first)
        return self.created_at > other.created_at

class NotificationSound(QObject):
    """Handles sound effects for notifications."""
    
    def __init__(self, sound_dir: str = None, parent=None):
        super().__init__(parent)
        # Set default sound directory if not provided
        if sound_dir is None:
            # Try to locate sounds directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sound_dir = os.path.join(current_dir, "..", "..", "resources", "sounds")
            
            # Check if directory exists, otherwise use current directory
            if not os.path.exists(sound_dir):
                sound_dir = current_dir
                
        self.sound_dir = sound_dir
        self.sounds = {}
        self.enabled = True
        self.volume = 0.8  # 0.0 - 1.0
        self._load_sounds()
        
    def _load_sounds(self):
        """Load sound files for different notification types."""
        for notification_type in NotificationType:
            sound_file = notification_type.value["sound"]
            sound_path = os.path.join(self.sound_dir, sound_file)
            
            # Only load if file exists
            if os.path.exists(sound_path):
                self.sounds[notification_type] = sound_path
            else:
                print(f"Warning: Sound file not found: {sound_path}")
                
    def play(self, notification_type: NotificationType):
        """Play sound for notification type."""
        if not self.enabled:
            return
            
        if notification_type in self.sounds:
            try:
                QSound.play(self.sounds[notification_type])
            except Exception as e:
                print(f"Error playing sound: {e}")
                
    def set_enabled(self, enabled: bool):
        """Enable or disable sounds."""
        self.enabled = enabled
        
    def set_volume(self, volume: float):
        """Set volume level (0.0 - 1.0)."""
        self.volume = max(0.0, min(1.0, volume))

class NotificationItemWidget(QFrame):
    """
    Widget for displaying a single notification.
    
    Signals:
        dismissed: Emitted when notification is dismissed
        action_triggered: Emitted when an action is triggered
    """
    
    dismissed = pyqtSignal(str)  # Notification ID
    action_triggered = pyqtSignal(str, int)  # Notification ID, Action index
    
    def __init__(self, notification: Notification, parent=None):
        super().__init__(parent)
        self.notification = notification
        self.hover = False
        
        # Setup UI
        self._init_ui()
        
        # Setup auto-dismiss timer if needed
        if notification.duration > 0:
            self.dismiss_timer = QTimer(self)
            self.dismiss_timer.setSingleShot(True)
            self.dismiss_timer.timeout.connect(self._auto_dismiss)
            self.dismiss_timer.start(notification.duration)
            
        # Track mouse for hover effects
        self.setMouseTracking(True)
        
    def _init_ui(self):
        """Initialize UI components."""
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        
        # Configure colors based on notification type
        color = QColor(self.notification.type.value["color"])
        lighter_color = QColor(color.lighter(115))
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Background, lighter_color)
        self.setPalette(palette)
        
        # Set border style
        self.setStyleSheet(f"""
            NotificationItemWidget {{
                border: 1px solid {color.name()};
                border-left: 4px solid {color.name()};
                border-radius: 4px;
                background-color: {lighter_color.name()};
                margin: 2px;
            }}
            NotificationItemWidget:hover {{
                background-color: {lighter_color.lighter(105).name()};
            }}
            QLabel {{
                background: transparent;
            }}
            QPushButton {{
                background-color: transparent;
                border: 1px solid {color.name()};
                border-radius: 3px;
                padding: 3px 8px;
                color: {color.darker(150).name()};
            }}
            QPushButton:hover {{
                background-color: {color.name()};
                color: white;
            }}
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header layout with title and close button
        header_layout = QHBoxLayout()
        
        # Load icon if available
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "icons", self.notification.type.value["icon"]
        )
        
        # Add icon if file exists
        if os.path.exists(icon_path):
            icon_label = QLabel()
            pixmap = QPixmap(icon_path).scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            header_layout.addWidget(icon_label)
        
        # Title label
        title_label = QLabel(self.notification.title)
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label, 1)  # Stretch
        
        # Timestamp label
        time_str = self.notification.created_at.strftime("%H:%M:%S")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: rgba(0, 0, 0, 150); font-size: 9pt;")
        header_layout.addWidget(time_label)
        
        # Close button if dismissible
        if self.notification.dismissible:
            close_btn = QPushButton("×")
            close_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 0px;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    color: darkred;
                }
            """)
            close_btn.clicked.connect(self._dismiss)
            header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Message
        message_label = QLabel(self.notification.message)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)
        
        # Progress bar if applicable
        if self.notification.progress is not None:
            progress_layout = QHBoxLayout()
            progress_layout.setContentsMargins(0, 5, 0, 5)
            
            # Progress background
            progress_bg = QFrame()
            progress_bg.setFixedHeight(6)
            progress_bg.setStyleSheet(f"""
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 3px;
            """)
            
            # Progress indicator
            progress_value = max(0, min(100, self.notification.progress))
            progress_indicator = QFrame(progress_bg)
            progress_indicator.setFixedHeight(6)
            progress_indicator.setFixedWidth(int(progress_bg.width() * progress_value / 100))
            progress_indicator.setStyleSheet(f"""
                background-color: {color.name()};
                border-radius: 3px;
            """)
            
            progress_layout.addWidget(progress_bg)
            layout.addLayout(progress_layout)
        
        # Action buttons
        if self.notification.actions:
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(0, 5, 0, 0)
            actions_layout.addStretch()
            
            for idx, action in enumerate(self.notification.actions):
                action_btn = QPushButton(action.text)
                action_btn.clicked.connect(lambda _, i=idx: self._action_triggered(i))
                actions_layout.addWidget(action_btn)
            
            layout.addLayout(actions_layout)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
    def _dismiss(self):
        """Handle dismissal of notification."""
        # Stop auto-dismiss timer if exists
        if hasattr(self, 'dismiss_timer') and self.dismiss_timer.isActive():
            self.dismiss_timer.stop()
            
        # Mark notification as dismissed
        self.notification.dismiss()
        
        # Create fade-out animation
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out.setDuration(200)
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_out.finished.connect(lambda: self.dismissed.emit(self.notification.id))
        self.fade_out.start()
        
    def _auto_dismiss(self):
        """Auto-dismiss when timer expires."""
        self._dismiss()
        
    def _action_triggered(self, action_idx):
        """Handle action button click."""
        # Trigger the action callback
        self.notification.actions[action_idx].trigger()
        
        # Emit signal
        self.action_triggered.emit(self.notification.id, action_idx)
        
    def enterEvent(self, event):
        """Handle mouse enter event."""
        self.hover = True
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave event."""
        self.hover = False
        self.update()
        super().leaveEvent(event)
        
    def resizeEvent(self, event):
        """Handle resize event for proper progress bar sizing."""
        super().resizeEvent(event)
        
        # Update progress indicator width if exists
        if self.notification.progress is not None:
            progress_layout = self.layout().itemAt(2)
            if progress_layout and progress_layout.count() > 0:
                progress_bg = progress_layout.itemAt(0).widget()
                if progress_bg:
                    progress_value = max(0, min(100, self.notification.progress))
                    progress_indicator = progress_bg.findChild(QFrame)
                    if progress_indicator:
                        progress_indicator.setFixedWidth(int(progress_bg.width() * progress_value / 100))

class NotificationManager(QObject):
    """
    Manages notification creation, queueing, and lifecycle.
    
    Signals:
        notification_added: Emitted when a notification is added
        notification_removed: Emitted when a notification is removed
        notifications_cleared: Emitted when all notifications are cleared
    """
    
    notification_added = pyqtSignal(object)
    notification_removed = pyqtSignal(str)
    notifications_cleared = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notifications = {}
        self.queue = queue.PriorityQueue()
        self.sound_manager = NotificationSound()
        
        # Maximum notifications to show
        self.max_visible = 5
        
        # Settings
        self.settings = {
            'sound_enabled': True,
            'notification_limit': 100,  # Max stored notifications
            'auto_dismiss_enabled': True
        }
        
    def add_notification(
        self,
        notification_type: Union[NotificationType, str],
        message: str,
        title: str = None,
        duration: int = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        actions: List[NotificationAction] = None,
        sound_enabled: bool = None,
        data: Any = None,
        dismissible: bool = True,
        progress: int = None
    ) -> str:
        """Add a notification to the manager."""
        # Convert string type to enum if needed
        if isinstance(notification_type, str):
            try:
                notification_type = NotificationType[notification_type.upper()]
            except KeyError:
                notification_type = NotificationType.INFO
        
        # Override sound setting if specified
        if sound_enabled is None:
            sound_enabled = self.settings['sound_enabled']
        
        # Override auto-dismiss if disabled in settings
        if (duration is None or duration > 0) and not self.settings['auto_dismiss_enabled']:
            duration = 0
            
        # Create notification
        notification = Notification(
            type=notification_type,
            message=message,
            title=title,
            duration=duration,
            priority=priority,
            actions=actions,
            sound_enabled=sound_enabled,
            data=data,
            dismissible=dismissible,
            progress=progress
        )
        
        # Store notification
        self.notifications[notification.id] = notification
        self.queue.put(notification)
        
        # Play sound if enabled
        if sound_enabled:
            self.sound_manager.play(notification_type)
            
        # Emit signal
        self.notification_added.emit(notification)
        
        # Enforce notification limit
        self._enforce_notification_limit()
        
        return notification.id
        
    def remove_notification(self, notification_id: str) -> bool:
        """Remove a notification from the manager."""
        if notification_id in self.notifications:
            # Mark as dismissed
            self.notifications[notification_id].dismiss()
            
            # Remove from notifications dict
            del self.notifications[notification_id]
            
            # Emit signal
            self.notification_removed.emit(notification_id)
            
            return True
        return False
        
    def clear_all_notifications(self):
        """Clear all notifications."""
        # Mark all notifications as dismissed
        for notification in self.notifications.values():
            notification.dismiss()
            
        # Clear collections
        self.notifications.clear()
        
        # Clear the queue (not directly possible with PriorityQueue)
        # Create a new empty queue instead
        self.queue = queue.PriorityQueue()
        
        # Emit signal
        self.notifications_cleared.emit()
        
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get a notification by ID."""
        return self.notifications.get(notification_id)
        
    def get_all_notifications(self) -> List[Notification]:
        """Get all notifications."""
        return list(self.notifications.values())
        
    def get_notifications_by_type(self, notification_type: NotificationType) -> List[Notification]:
        """Get notifications by type."""
        return [n for n in self.notifications.values() if n.type == notification_type]
        
    def get_unread_notifications(self) -> List[Notification]:
        """Get unread notifications."""
        return [n for n in self.notifications.values() if not n.read]
        
    def update_notification_progress(self, notification_id: str, progress: int) -> bool:
        """Update progress of a notification."""
        if notification_id in self.notifications:
            self.notifications[notification_id].progress = max(0, min(100, progress))
            return True
        return False
        
    def update_notification_message(self, notification_id: str, message: str) -> bool:
        """Update message of a notification."""
        if notification_id in self.notifications:
            self.notifications[notification_id].message = message
            return True
        return False
        
    def update_notification_title(self, notification_id: str, title: str) -> bool:
        """Update title of a notification."""
        if notification_id in self.notifications:
            self.notifications[notification_id].title = title
            return True
        return False
        
    def mark_notification_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        if notification_id in self.notifications:
            self.notifications[notification_id].mark_as_read()
            return True
        return False
        
    def get_next_notification(self) -> Optional[Notification]:
        """Get the next notification from the queue."""
        if not self.queue.empty():
            return self.queue.get_nowait()
        return None
        
    def save_notifications(self, file_path: str) -> bool:
        """Save notifications to file."""
        try:
            # Convert notifications to serializable format
            serializable = []
            for notification in self.notifications.values():
                serializable.append({
                    'id': notification.id,
                    'type': notification.type.name,
                    'title': notification.title,
                    'message': notification.message,
                    'created_at': notification.created_at.isoformat(),
                    'priority': notification.priority.name,
                    'read': notification.read,
                    'dismissed': notification.dismissed,
                    'data': notification.data if isinstance(notification.data, (str, int, float, bool, type(None))) else str(notification.data)
                })
                
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(serializable, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving notifications: {e}")
            return False
            
    def load_notifications(self, file_path: str) -> bool:
        """Load notifications from file."""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Clear existing notifications
            self.clear_all_notifications()
            
            # Read from file
            with open(file_path, 'r') as f:
                serialized = json.load(f)
                
            # Convert to notifications
            for item in serialized:
                # Skip if already dismissed
                if item.get('dismissed', False):
                    continue
                    
                # Create notification
                notification_type = NotificationType[item['type']]
                notification = Notification(
                    type=notification_type,
                    message=item['message'],
                    title=item['title'],
                    duration=0,  # Don't auto-dismiss loaded notifications
                    priority=NotificationPriority[item['priority']],
                    data=item.get('data')
                )
                
                # Set ID and created_at
                notification.id = item['id']
                notification.created_at = datetime.fromisoformat(item['created_at'])
                
                # Set read state
                if item.get('read', False):
                    notification.mark_as_read()
                    
                # Add to collections
                self.notifications[notification.id] = notification
                self.queue.put(notification)
                
            return True
        except Exception as e:
            print(f"Error loading notifications: {e}")
            return False
            
    def set_sound_enabled(self, enabled: bool):
        """Enable or disable sounds."""
        self.settings['sound_enabled'] = enabled
        self.sound_manager.set_enabled(enabled)
        
    def set_auto_dismiss_enabled(self, enabled: bool):
        """Enable or disable auto-dismissal."""
        self.settings['auto_dismiss_enabled'] = enabled
        
    def set_sound_volume(self, volume: float):
        """Set sound volume level."""
        self.sound_manager.set_volume(volume)
        
    def _enforce_notification_limit(self):
        """Enforce the maximum number of stored notifications."""
        # If we're over the limit, remove oldest notifications first
        if len(self.notifications) > self.settings['notification_limit']:
            # Sort by creation time (oldest first)
            sorted_notifications = sorted(
                self.notifications.values(),
                key=lambda n: n.created_at
            )
            
            # Remove excess notifications
            excess_count = len(self.notifications) - self.settings['notification_limit']
            for i in range(excess_count):
                notification_id = sorted_notifications[i].id
                self.remove_notification(notification_id)

class NotificationWidget(QWidget):
    """
    Widget for displaying and managing notifications.
    
    Signals:
        notification_dismissed: Emitted when a notification is dismissed
        action_triggered: Emitted when a notification action is triggered
    """
    
    notification_dismissed = pyqtSignal(str)
    action_triggered = pyqtSignal(str, int, object)  # notification_id, action_index, notification_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create notification manager
        self.notification_manager = NotificationManager()
        
        # Connect signals
        self.notification_manager.notification_added.connect(self._on_notification_added)
        self.notification_manager.notification_removed.connect(self._on_notification_removed)
        self.notification_manager.notifications_cleared.connect(self._on_notifications_cleared)
        
        # Setup UI
        self._init_ui()
        
        # Set initial size
        self.setMinimumWidth(300)
        self.setMaximumWidth(500)
        
    def _init_ui(self):
        """Initialize UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scroll area for notifications
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Notification container
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(5)
        self.container_layout.addStretch()
        
        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)
        
        # Empty state label
        self.empty_label = QLabel("No notifications")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: gray; padding: 20px;")
        self.container_layout.insertWidget(0, self.empty_label)
        
        # Hide empty label if we have notifications
        self.empty_label.setVisible(len(self.notification_manager.notifications) == 0)
        
    def add_notification(
        self,
        notification_type: Union[NotificationType, str],
        message: str,
        title: str = None,
        duration: int = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        actions: List[NotificationAction] = None,
        sound_enabled: bool = None,
        data: Any = None,
        dismissible: bool = True,
        progress: int = None
    ) -> str:
        """Add a notification through the notification manager."""
        notification_id = self.notification_manager.add_notification(
            notification_type=notification_type,
            message=message,
            title=title,
            duration=duration,
            priority=priority,
            actions=actions,
            sound_enabled=sound_enabled,
            data=data,
            dismissible=dismissible,
            progress=progress
        )
        
        return notification_id
        
    def remove_notification(self, notification_id: str) -> bool:
        """Remove a notification through the notification manager."""
        return self.notification_manager.remove_notification(notification_id)
        
    def clear_all_notifications(self):
        """Clear all notifications through the notification manager."""
        self.notification_manager.clear_all_notifications()
        
    def update_notification_progress(self, notification_id: str, progress: int) -> bool:
        """Update progress of a notification."""
        # Update in manager
        success = self.notification_manager.update_notification_progress(notification_id, progress)
        
        # Update UI
        if success:
            # Find the notification widget
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if isinstance(widget, NotificationItemWidget) and widget.notification.id == notification_id:
                    widget.notification.progress = progress
                    widget.update()
                    break
                    
        return success
        
    def _on_notification_added(self, notification):
        """Handle notification added event."""
        # Hide empty label
        self.empty_label.setVisible(False)
        
        # Create notification widget
        notification_widget = NotificationItemWidget(notification)
        notification_widget.dismissed.connect(self._on_item_dismissed)
        notification_widget.action_triggered.connect(self._on_item_action_triggered)
        
        # Limit visible notifications
        visible_count = sum(1 for i in range(self.container_layout.count())
                         if isinstance(self.container_layout.itemAt(i).widget(), NotificationItemWidget))
                         
        # If we hit the limit, remove the oldest one
        if visible_count >= self.notification_manager.max_visible:
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if isinstance(widget, NotificationItemWidget):
                    self.container_layout.removeWidget(widget)
                    widget.deleteLater()
                    break
        
        # Add the new notification at the top
        self.container_layout.insertWidget(0, notification_widget)
        
        # Create fade-in animation
        notification_widget.setVisible(False)
        
        # Use QTimer to ensure widget is properly initialized before animation
        QTimer.singleShot(50, lambda: self._animate_notification_in(notification_widget))
        
    def _animate_notification_in(self, widget):
        """Animate a notification widget appearing."""
        widget.setVisible(True)
        
        # Create opacity effect
        opacity_effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(opacity_effect)
        
        # Create animation
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(250)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        
    def _on_notification_removed(self, notification_id):
        """Handle notification removed event."""
        # Find and remove the notification widget
        for i in range(self.container_layout.count()):
            widget = self.container_layout.itemAt(i).widget()
            if isinstance(widget, NotificationItemWidget) and widget.notification.id == notification_id:
                self.container_layout.removeWidget(widget)
                widget.deleteLater()
                break
                
        # Show empty label if no notifications left
        has_notifications = False
        for i in range(self.container_layout.count()):
            widget = self.container_layout.itemAt(i).widget()
            if isinstance(widget, NotificationItemWidget):
                has_notifications = True
                break
                
        self.empty_label.setVisible(not has_notifications)
        
        # Emit signal
        self.notification_dismissed.emit(notification_id)
        
    def _on_notifications_cleared(self):
        """Handle all notifications cleared event."""
        # Remove all notification widgets
        for i in range(self.container_layout.count() - 1, -1, -1):
            widget = self.container_layout.itemAt(i).widget()
            if isinstance(widget, NotificationItemWidget):
                self.container_layout.removeWidget(widget)
                widget.deleteLater()
                
        # Show empty label
        self.empty_label.setVisible(True)
        
    def _on_item_dismissed(self, notification_id):
        """Handle notification item dismissed event."""
        self.notification_manager.remove_notification(notification_id)
        
    def _on_item_action_triggered(self, notification_id, action_idx):
        """Handle notification item action triggered event."""
        notification = self.notification_manager.get_notification(notification_id)
        if notification:
            # Emit signal with notification data
            self.action_triggered.emit(notification_id, action_idx, notification.data)
            
    def set_sound_enabled(self, enabled: bool):
        """Enable or disable sounds."""
        self.notification_manager.set_sound_enabled(enabled)
        
    def set_auto_dismiss_enabled(self, enabled: bool):
        """Enable or disable auto-dismissal."""
        self.notification_manager.set_auto_dismiss_enabled(enabled)
        
    def set_sound_volume(self, volume: float):
        """Set sound volume level."""
        self.notification_manager.set_sound_volume(volume)
        
    def set_max_visible(self, max_visible: int):
        """Set maximum visible notifications."""
        self.notification_manager.max_visible = max_visible
        
    def get_notification_count(self) -> int:
        """Get number of active notifications."""
        return len(self.notification_manager.notifications)


# Example usage in a main application window
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QDockWidget, QTextEdit, QPushButton
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("Notification Widget Demo")
            self.setGeometry(100, 100, 800, 600)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Create layout
            layout = QVBoxLayout(central_widget)
            
            # Add text area
            self.text_area = QTextEdit()
            self.text_area.setText("This is a demo of the notification widget.")
            layout.addWidget(self.text_area)
            
            # Add test buttons
            buttons_layout = QHBoxLayout()
            
            success_btn = QPushButton("Success")
            success_btn.clicked.connect(lambda: self.add_test_notification(NotificationType.SUCCESS))
            buttons_layout.addWidget(success_btn)
            
            info_btn = QPushButton("Info")
            info_btn.clicked.connect(lambda: self.add_test_notification(NotificationType.INFO))
            buttons_layout.addWidget(info_btn)
            
            warning_btn = QPushButton("Warning")
            warning_btn.clicked.connect(lambda: self.add_test_notification(NotificationType.WARNING))
            buttons_layout.addWidget(warning_btn)
            
            error_btn = QPushButton("Error")
            error_btn.clicked.connect(lambda: self.add_test_notification(NotificationType.ERROR))
            buttons_layout.addWidget(error_btn)
            
            trade_btn = QPushButton("Trade")
            trade_btn.clicked.connect(lambda: self.add_test_notification(NotificationType.TRADE))
            buttons_layout.addWidget(trade_btn)
            
            layout.addLayout(buttons_layout)
            
            # Create notification dock widget
            self.notification_dock = QDockWidget("Notifications", self)
            self.notification_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            
            # Create notification widget
            self.notification_widget = NotificationWidget()
            self.notification_dock.setWidget(self.notification_widget)
            
            # Add dock widget to main window
            self.addDockWidget(Qt.RightDockWidgetArea, self.notification_dock)
            
            # Connect signals
            self.notification_widget.notification_dismissed.connect(self.on_notification_dismissed)
            self.notification_widget.action_triggered.connect(self.on_action_triggered)
            
        def add_test_notification(self, notification_type):
            """Add a test notification of the specified type."""
            # Create different messages based on type
            if notification_type == NotificationType.SUCCESS:
                message = "Operation completed successfully."
                actions = [NotificationAction("Details", self.show_details, "Success details")]
            elif notification_type == NotificationType.INFO:
                message = "System is running normally."
                actions = None
            elif notification_type == NotificationType.WARNING:
                message = "Approaching memory limit. Consider closing unused applications."
                actions = [
                    NotificationAction("Ignore", self.ignore_warning),
                    NotificationAction("Optimize", self.optimize_memory)
                ]
            elif notification_type == NotificationType.ERROR:
                message = "Failed to connect to server. Check your network connection."
                actions = [NotificationAction("Retry", self.retry_connection)]
            elif notification_type == NotificationType.TRADE:
                message = "BTC/USDT: Buy order executed at $48,250.00"
                actions = [NotificationAction("View", self.view_trade, {"symbol": "BTC/USDT", "price": 48250.00})]
            else:
                message = "Notification message."
                actions = None
                
            # Add notification
            notification_id = self.notification_widget.add_notification(
                notification_type=notification_type,
                message=message,
                title=None,  # Use default
                actions=actions
            )
            
            # Log to text area
            self.text_area.append(f"Added notification: {notification_id}, Type: {notification_type.name}")
            
        def on_notification_dismissed(self, notification_id):
            """Handle notification dismissed."""
            self.text_area.append(f"Notification dismissed: {notification_id}")
            
        def on_action_triggered(self, notification_id, action_idx, data):
            """Handle notification action triggered."""
            self.text_area.append(f"Action triggered: {notification_id}, Action: {action_idx}, Data: {data}")
            
        def show_details(self, details):
            """Show details action."""
            self.text_area.append(f"Showing details: {details}")
            
        def ignore_warning(self):
            """Ignore warning action."""
            self.text_area.append("Warning ignored.")
            
        def optimize_memory(self):
            """Optimize memory action."""
            self.text_area.append("Optimizing memory...")
            
        def retry_connection(self):
            """Retry connection action."""
            self.text_area.append("Retrying connection...")
            
        def view_trade(self, trade_data):
            """View trade action."""
            self.text_area.append(f"Viewing trade: {trade_data}")
            
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
