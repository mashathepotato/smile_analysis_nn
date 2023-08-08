import sqlite3
import json
import datetime
import random


def get_routine_of_the_day():
    conn = sqlite3.connect("userdata.db")
    cur = conn.cursor()

    today = datetime.date.today()
    cur.execute("SELECT * FROM routines WHERE date=?", (today,))
    result = cur.fetchone()

    conn.close()

    if result:
        routine = {
            "id": result[0],
            "name": result[1],
            "data": json.loads(result[2]),
            "date": result[3]
        }
        return routine
    else:
        return None
    
def add_product_to_routine(product_type):
    routine = get_routine_of_the_day()

    if routine:
        new_product = {"name": product_type, "state": "unchecked"}
        routine["data"].append(new_product)

        conn = sqlite3.connect("userdata.db")
        cur = conn.cursor()

        cur.execute("UPDATE routines SET data=? WHERE id=?", (json.dumps(routine["data"]), routine["id"]))

        conn.commit()
        conn.close()

        print("Product added to the routine.")
    else:
        print("No routine found for today.")