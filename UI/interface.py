import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QIcon, QFont, QPixmap
# from code import main
import subprocess

class CustomWidget(QWidget):
    def __init__(self):
        super().__init__()

        mirror_background = QWidget(self)
        mirror_background.setStyleSheet("background-color: silver;")
        # mirror_background.setGeometry(0, 0, 1920, 1080)
        mirror_background.setGeometry(0, 0, 2260, 1800)

        # Create a grid layout to hold the buttons
        layout = QGridLayout()

        # Create four buttons and add them to the layout
        button1 = self.create_button("Analytics", "UI/icons/analytics.png")
        button2 = self.create_button("Skin Analysis", "UI/icons/camera.png")
        button3 = self.create_button("Routine", "UI/icons/routine.png")
        button4 = self.create_button("Smile Analysis", "UI/icons/tooth2.png")

        layout.addWidget(button1, 0, 0)
        layout.addWidget(button2, 1, 0)
        layout.addWidget(button3, 0, 1)
        layout.addWidget(button4, 1, 1)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set the window properties
        self.setWindowTitle("Custom Icon Buttons")
        self.setGeometry(0, 0, 1920, 1080)  # Full-screen dimensions (adjust as needed)

    def create_button(self, text, icon_path):
        button = QPushButton(text)
        button_font = QFont(" Roboto", 44)
        button.setFont(button_font)
        icon = QIcon(icon_path)
        button.setIcon(icon)
        button.setIconSize(button.sizeHint())  # Set the icon size to match the button size
        button.setFixedSize(1000, 600)  # Set the button size to be square and larger
        button.clicked.connect(self.custom_function)
        # button.setStyleSheet("background-color: purple; color: white;")
        # button.setStyleSheet("QPushButton:pressed {background-color: lightpurple;}")
        button.setStyleSheet("""
            background-color: purple;
            color: white;
            border: 1px solid darkpurple;
        """)

        # Add style for button pressed state (light up effect)
        # button.setStyleSheet("""
        #     QPushButton:pressed {
        #         background-color: lightpurple;
        #     }
        # """)

        return button

    def custom_function(self):
        sender = self.sender()  # Get the button that triggered the event
        print(f"{sender.text()} was pressed")

        if sender.text() == "Smile Analysis":
            try:
                subprocess.Popen(["python", "main.py"])
                sys.exit(app.exec())
            except Exception as e:
                print(f"Error running main.py: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CustomWidget()
    window.showFullScreen()  # Show the window in full-screen mode
    sys.exit(app.exec_())


# Paths to change: main img, sequential train/test