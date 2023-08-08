from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QSpacerItem, QSizePolicy, QMessageBox, QStyle
from PyQt5.QtCore import Qt, QSize
import sqlite3
import json

from apps.BaseApp import BaseApp

import pyqtgraph as pg
import numpy as np

class AnalyticsApp(BaseApp):
    def __init__(self, parent=None):
        super(AnalyticsApp, self).__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Title
        title = QLabel("Statistics")
        title.setFont(QFont("Arial", 24))
        title.setStyleSheet("color: white;")
        layout.addWidget(title, alignment=Qt.AlignHCenter)

        routines = self.load_routines_from_db()

        y_vals = []
        for routine in routines:
            # count the number of checked products in the routine
            y_vals.append(len([x for x in routine["data"] if x["state"]=="checked"]))

        #y_vals = y_vals.reverse()

        # Bar graph
        bar_graph = pg.PlotWidget()
        y_vals = np.array(y_vals)
        x_vals = np.arange(len(y_vals))
        bg = pg.BarGraphItem(x=x_vals, height=y_vals, width=0.6, brush="w")
        bar_graph.addItem(bg)
        layout.addWidget(bar_graph)

        # Line graph
        line_graph = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(size=len(x)) * 0.1
        line_graph.plot(x, y, pen="w")
        layout.addWidget(line_graph)

        self.app_frame.setLayout(layout)

    def load_routines_from_db(self):
        conn = sqlite3.connect("userdata.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM routines")
        rows = cur.fetchall()

        routines = [{"name": row["name"], "data": json.loads(row["data"]), "date":row["date"]} for row in rows]

        conn.close()
        return routines