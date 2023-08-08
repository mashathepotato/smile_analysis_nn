import sqlite3
import json
import datetime
import random

def createFakeRoutine(products_count):
    product_types = ["Moisturiser",
        "Cleanser",
        "Exfoliator",
        "Face Mask",
        "Toner",
        "Sun Cream",
        "Night Cream"]
    states = ["checked","unchecked"]

    fake_data = []
    for i in range(products_count):
        fake_data.append({"name":random.choice(product_types), "state":random.choice(states)})

    return fake_data

def create_and_seed_db():

    conn = sqlite3.connect("userdata.db")
    cur = conn.cursor()

    # Create the products table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data JSON NOT NULL, 
            date DATE NOT NULL
        )
    """)

    fake_days = 20

    for i in range(fake_days):
        routine = {
            "name": "morning",
            "data": createFakeRoutine(products_count=5), 
            "date": datetime.date.today()+datetime.timedelta(days=-i)
        }
        cur.execute("INSERT INTO routines (name, data, date) VALUES (?, ?, ?)", (routine["name"], json.dumps(routine["data"]), routine["date"]))

    conn.commit()
    conn.close()

create_and_seed_db()
