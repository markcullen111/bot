## performance_tab.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class PerformanceTab(QWidget):
    """Tab for monitoring system performance."""
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Performance Metrics"))
        
        self.setLayout(layout)

