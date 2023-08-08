
from PyQt5.QtWidgets import QPushButton, QLabel
from PyQt5.QtGui import QFont

def createButton(text, size=32, color="purple"):
    button = QPushButton(text)
    button.setFont(QFont("Arial", size))
    button.setStyleSheet("background-color: "+color+"; color: white; padding: 40px; border-radius: 15px;")
    return button

def createTitle(text, size=32):
    title = QLabel(text)
    title.setFont(QFont("Arial", size))
    title.setStyleSheet("color: white; background-color: none;")
    return title