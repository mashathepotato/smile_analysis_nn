import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QSpacerItem, QSizePolicy, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QFont, QIcon
import gui_elements
import database_queries
#from demo.recommender import *

from apps.BaseApp import BaseApp
from PyQt5.QtCore import Qt, QSize

from PyQt5.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    frame_ready = pyqtSignal(QPixmap)

    def run(self):
        self.capture = cv2.VideoCapture(0)
        while True:
            ret, frame = self.capture.read()

            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.Canny(frame, 100, 200)  # Perform edge detection
            h, w = frame.shape
            colored = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            qimage = QImage(colored.data, w, h, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.frame_ready.emit(pixmap)

    def get_picture(self):
        ret, frame = self.capture.read()
        return frame

    def stop(self):
        self.capture.release()


class OpenCVApp(BaseApp):
    def __init__(self, parent=None):
        super(OpenCVApp, self).__init__(parent)
        #self.recommender = recommender()
        self.setup_ui()


    def setup_ui(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(10)
        bottom_layout = QHBoxLayout()

        # Add title
        top_layout.addWidget(gui_elements.createTitle("Analysis", size=42), Qt.AlignHCenter)

        # Add product recommendation
        recommender_layout = QVBoxLayout()
        recommender_layout.setContentsMargins(20, 20, 20, 20)
        recommender_layout.setSpacing(10)

        recommender_layout.addWidget(gui_elements.createTitle("Recommendation Types:"))

        product_types = ["Moisturiser",
        "Cleanser",
        "Exfoliator",
        "Face Mask",
        "Toner",
        "Sun Cream",
        "Night Cream"]

        button1 = gui_elements.createButton("Moisturiser")
        button1.clicked.connect(lambda: self.recommend_clicked(button1.text()))
        recommender_layout.addWidget(button1)

        button2 = gui_elements.createButton("Cleanser")
        button2.clicked.connect(lambda: self.recommend_clicked(button2.text()))
        recommender_layout.addWidget(button2)

        button3 = gui_elements.createButton("Exfoliator")
        button3.clicked.connect(lambda: self.recommend_clicked(button3.text()))
        recommender_layout.addWidget(button3)

        pictureButton = gui_elements.createButton(" Analyse Face", color="green")
        pictureButton.clicked.connect(self.analyse_clicked)
        pictureButton.setIcon(QIcon("icons/camera.png"))
        pictureButton.setIconSize(QSize(50,50))
        recommender_layout.addWidget(pictureButton)

        self.feedback = gui_elements.createTitle("Ready to recommend!")
        recommender_layout.addWidget(self.feedback)

        bottom_layout.addLayout(recommender_layout)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(10)

        self.camera_feed_label = QLabel()
        self.camera_feed_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(self.camera_feed_label)

        routineButton = gui_elements.createButton("Add Recommendation To Routine")
        routineButton.clicked.connect(lambda: self.routine_add_clicked(self.feedback.text()))
        recommender_layout.addWidget(routineButton)


        bottom_layout.addLayout(right_layout)


        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.camera_feed_label.setPixmap)
        self.camera_thread.start()

        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

        self.app_frame.setLayout(layout)

    def recommend_clicked(self, product):
        #recommendations = self.recommender.recommend(product_type)
        self.feedback.setText("We recommend: Nivea "+product)
        self.recommended_product = product
        pass

    def analyse_clicked(self):
        image = self.camera_thread.capture()
        #self.recommender.analyse(image)

    def routine_add_clicked(self, product):
        self.feedback.setText("Added "+self.recommended_product+" to Routine!")
        database_queries.add_product_to_routine(product)

    

    def stop_camera_and_show_store_screen(self):
        self.camera_thread.stop()
        self.camera_thread.wait()
        self.parent().parent().show_store_screen()
