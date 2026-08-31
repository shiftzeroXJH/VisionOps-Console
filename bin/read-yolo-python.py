import sqlite3
import sys


database_path = sys.argv[1]
with sqlite3.connect(database_path) as connection:
    row = connection.execute(
        "SELECT value FROM settings WHERE key = ?", ("yolo_python",)
    ).fetchone()
if row and row[0]:
    print(str(row[0]).strip())
