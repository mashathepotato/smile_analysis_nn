from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QGridLayout, QFrame
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QSpacerItem, QSizePolicy
import cv2

from apps.RoutineApp import RoutineApp
from apps.OpenCVApp import OpenCVApp
from apps.AnalyticsApp import AnalyticsApp


class StandbyScreen(QWidget):
    def __init__(self, parent=None):
        super(StandbyScreen, self).__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        start_button = QPushButton("Start")
        start_button.setFont(QFont("Arial", 30))
        start_button.setStyleSheet("background-color: purple; color: white; padding: 40px; border-radius: 15px;")
        layout.addWidget(start_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)
        start_button.clicked.connect(self.parent().show_store_screen)

class StoreScreen(QWidget):
    def __init__(self, parent=None):
        super(StoreScreen, self).__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()

        self.grid_layout = QGridLayout()

        grid_frame = QFrame()
        grid_frame.setLayout(self.grid_layout)

        self.layout.addWidget(grid_frame)

        back_button = QPushButton(" Back")
        back_button.setFont(QFont("Arial", 32))
        back_button.setStyleSheet("background-color: purple; color: white; padding: 40px; border-radius: 15px;")
        back_button.clicked.connect(self.parent().show_standby_screen)
        back_button.setIcon(QIcon("icons/back.png"))
        back_button.setIconSize(QSize(50,50))
        
        self.layout.addWidget(back_button)

        self.setLayout(self.layout)

    def register_app(self, row, col, title, icon_path, show_app_callback):
        app_frame = QFrame()
        app_frame.setFrameShape(QFrame.StyledPanel)
        app_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 0, 255, 80);
                border-radius: 15px;
            }
        """)
        title_layout = QHBoxLayout()

        app_icon = QLabel()
        app_icon.setPixmap(QPixmap(icon_path))
        app_icon.setStyleSheet("background-color: none; color: white;")
        title_layout.addWidget(app_icon)

        app_label = QLabel(title)
        app_label.setFont(QFont("Arial", 32))
        app_label.setStyleSheet("background-color: none; color: white; padding: 40px;")
        title_layout.addWidget(app_label)


        app_frame.setLayout(title_layout)
        app_frame.mousePressEvent = lambda event: show_app_callback()

        self.grid_layout.addWidget(app_frame, row, col)
        

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('Smart Mirror')
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowState(Qt.WindowFullScreen)
        self.setStyleSheet("background-color: black;")
        
        self.stacked_widget = QStackedWidget(self)
        self.standby_screen = StandbyScreen(self)
        self.store_screen = StoreScreen(self)
        
        self.stacked_widget.addWidget(self.standby_screen)
        self.stacked_widget.addWidget(self.store_screen)
        self.setCentralWidget(self.stacked_widget)

        self.routine_app = RoutineApp(self)
        self.stacked_widget.addWidget(self.routine_app)
        self.store_screen.register_app(0, 0, "Morning Routine", "icons/routine.png", self.show_routine_app)

        self.open_cv_app = OpenCVApp(self)
        self.stacked_widget.addWidget(self.open_cv_app)
        self.store_screen.register_app(0, 1, "Analyse skin", "icons/camera.png", self.show_open_cv_app)

        self.analytics_app = AnalyticsApp(self)
        self.stacked_widget.addWidget(self.analytics_app)
        self.store_screen.register_app(1, 0, "Statistics", "icons/analytics.png", self.show_analytics_app)

        

    def show_standby_screen(self):
        self.stacked_widget.setCurrentWidget(self.standby_screen)
        
    def show_store_screen(self):
        self.stacked_widget.setCurrentWidget(self.store_screen)

    def show_routine_app(self):
        self.stacked_widget.setCurrentWidget(self.routine_app)

    def show_open_cv_app(self):
        self.stacked_widget.setCurrentWidget(self.open_cv_app)

    def show_analytics_app(self):
        self.stacked_widget.setCurrentWidget(self.analytics_app)

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
