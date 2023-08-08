from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QGridLayout, QFrame

class BaseApp(QWidget):
    def __init__(self, parent=None):
        super(BaseApp, self).__init__(parent)
        self.setup_layout()
        
    def setup_layout(self):
        self.layout = QVBoxLayout()
        self.app_frame = QFrame()
        self.layout.addWidget(self.app_frame)

        back_button = QPushButton(" Back")
        back_button.setFont(QFont("Arial", 32))
        back_button.setStyleSheet("background-color: purple; color: white; padding: 40px; border-radius: 15px;")
        back_button.clicked.connect(self.parent().show_store_screen)
        back_button.setIcon(QIcon("icons/back.png"))
        back_button.setIconSize(QSize(50,50))
        self.layout.addWidget(back_button)

        self.setLayout(self.layout)

    def setup_ui(self):
        pass
        
    def start(self):
        pass
        
    def stop(self):
        pass

    def add_back_button(self, layout):
        back_button = QPushButton("Back")
        back_button.setFont(QFont("Arial", 16))
        back_button.setStyleSheet("background-color: purple; color: white; padding: 40px; border-radius: 15px;")
        back_button.clicked.connect(self.parent().show_store_screen)
        layout.addWidget(back_button)

