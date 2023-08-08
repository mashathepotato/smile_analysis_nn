from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QSpacerItem, QSizePolicy, QMessageBox, QStyle
from PyQt5.QtCore import Qt, QSize
import sqlite3
import json
import datetime
# from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import database_queries

from apps.BaseApp import BaseApp

from PyQt5.QtWidgets import QProxyStyle

class LargeCheckboxStyle(QProxyStyle):
    def pixelMetric(self, metric, option=None, widget=None):
        if metric == QStyle.PM_IndicatorWidth or metric == QStyle.PM_IndicatorHeight:
            return 30
        return super().pixelMetric(metric, option, widget)

class RoutineApp(BaseApp):
    def __init__(self, parent=None):
        super(RoutineApp, self).__init__(parent)
        self.setup_ui()
        self.parent().stacked_widget.currentChanged.connect(self.on_stacked_widget_changed)


    def setup_checkboxes(self, routine, layout):
        
        if(routine):
            for product in routine["data"]:
                checkbox = QCheckBox(product["name"])
                checkbox.setChecked(product["state"]=="checked")
                checkbox.setFont(QFont("Arial", 26))
                checkbox.setStyleSheet("color: white; background-color: purple; border-radius: 15px; padding: 10px; width: 200px;")
                checkbox.setGeometry(0, 0, 0, 0)
                #checkbox.setIconSize(QSize(100,40))
                checkbox.setStyle(LargeCheckboxStyle())
                #checkbox.setMinimumSize(100,100)
                checkbox.setLayoutDirection(Qt.RightToLeft)
                layout.addWidget(checkbox, alignment=Qt.AlignHCenter)
                self.checkboxes.append(checkbox)
        else:
            print("Error: routine was None")

    
    def setup_ui(self):
        

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(10)

        app_label = QLabel("Morning Routine")

        font1 = app_label.font()
        font1.setPointSize(32)

        app_label.setFont(font1)
        app_label.setStyleSheet("background-color: none; color: white; border-radius: 15px;")

        top_layout.addWidget(app_label, alignment=Qt.AlignHCenter)

        current_date = datetime.datetime.now()
        date_label = QLabel("Current date: "+current_date.strftime("%m/%d/%Y, %H:%M:%S"))
        date_label.setFont(font1)
        top_layout.addWidget(date_label, alignment=Qt.AlignHCenter)


        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(20, 20, 20, 20)
        bottom_layout.setSpacing(10)


        # Load routines from the SQLite database
        #routines = self.load_routines_from_db()

        routine = database_queries.get_routine_of_the_day()
        
        # if no routine exists for today
        if(not routine):
            routine = self.create_todays_routine()

        # Left side layout for checkboxes
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)


        self.checkboxes = []

        # for now, just grab first routine
        self.setup_checkboxes(routine, left_layout)

        # Add a Done button
        done_button = QPushButton("Done")
        done_button.setFont(QFont("Arial", 32))
        done_button.setStyleSheet("background-color: green; color: white; padding: 10px; width: 200px;border-radius: 15px;")
        done_button.clicked.connect(self.check_items_and_return)
        left_layout.addWidget(done_button, alignment=Qt.AlignHCenter)
        

         # Right side layout for the YouTube video
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # # Embed a YouTube video
        # youtube_video_url = "https://www.youtube.com/embed/yptop_8_wqo"  # Replace VIDEO_ID with the desired video ID
        # web_view = QWebEngineView()
        # web_view.setUrl(QUrl(youtube_video_url))
        # right_layout.addWidget(web_view)

        # Combine left and right layouts
        bottom_layout.addLayout(left_layout)
        bottom_layout.addLayout(right_layout)

        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

        self.app_frame.setLayout(layout)

    def create_todays_routine(self):
        routine = {
            "id": 0,
            "name": "morning",
            "data": [],
            "date": datetime.date.today()
        }
        
        return routine

    def load_routines_from_db(self):
        conn = sqlite3.connect("userdata.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM routines")
        rows = cur.fetchall()

        routines = [{"name": row["name"], "data": json.loads(row["data"]), "date": row["date"]} for row in rows]

        conn.close()
        return routines

    def check_items_and_return(self):
        all_checked = all([checkbox.isChecked() for checkbox in self.checkboxes])

        if all_checked:
            self.save_progress()
            self.parent().parent().show_store_screen()
        else:
            QMessageBox.warning(self, "Warning", "Please make sure all the checkboxes are ticked.")

    def on_stacked_widget_changed(self):
        pass


    def save_progress(self):
        conn = sqlite3.connect("userdata.db")
        cur = conn.cursor()

        routine = {
            "name": "morning",
            "data": [{"name":checkbox.text(),"state":"checked" if checkbox.isChecked() else "unchecked"} for checkbox in self.checkboxes], 
            "date": datetime.date.today()
        }
        
        cur.execute("INSERT INTO routines (name, data, date) VALUES (?, ?, ?)", (routine["name"], json.dumps(routine["data"]), routine["date"]))
        conn.commit()
        conn.close()

