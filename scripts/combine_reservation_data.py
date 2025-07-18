"""
Combine reservation data into one database.

Still need to clean up data before it goes in.
"""

# %%
import csv
from pathlib import Path
import sqlite3
import re

# Compile once outside the function
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def clean_date(value: str) -> str | None:
    """
    Validate and normalize ISO 8601 date strings (YYYY-MM-DD).
    Assumes dates are already in roughly correct shape.
    """
    if not value:
        return None
    value = value[:10]  # remove time if present
    return value if ISO_DATE_RE.match(value) else None


# %%
# Create a SQLite database to combine all reservation data
conn = sqlite3.connect("../data/reservations/reservations.db")
cur = conn.cursor()

cur.execute(
    """
CREATE TABLE IF NOT EXISTS reservations (
    historicalreservationid TEXT,
    agency TEXT,
    orgid INTEGER,
    parentlocation TEXT,
    parentlocationid INTEGER,
    park TEXT,
    facilityid TEXT,
    facilityzip INTEGER,
    facilitystate TEXT,
    facilitylongitude REAL,
    facilitylatitude REAL,
    startdate TEXT,
    enddate TEXT,
    orderdate TEXT,
    numberofpeople INTEGER
);
"""
)
conn.commit()

# %%
# Inserting data into the database
conn = sqlite3.connect("../data/reservations/reservations.db")
cur.execute("PRAGMA synchronous = OFF")
cur = conn.cursor()

# List of expected columns in the table (order matters!)
expected_columns = [
    "historicalreservationid",
    "agency",
    "orgid",
    "parentlocation",
    "parentlocationid",
    "park",
    "facilityid",
    "facilityzip",
    "facilitystate",
    "facilitylongitude",
    "facilitylatitude",
    "startdate",
    "enddate",
    "orderdate",
    "numberofpeople",
]

csv_dir = Path("../data/reservations")
csv_files = csv_dir.glob("*.csv")


def count_lines(filepath):
    """ For checking file import progress. """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fi:
        return sum(1 for _ in fi)


for file_path in csv_files:
    lines = count_lines(file_path)
    print(f"Processing {file_path.name}...")
    BATCH_COUNT = 1
    with file_path.open("r", newline="", encoding="latin-1") as f:
        # Read and clean the header row
        raw_header_line = f.readline()
        raw_headers = [h.strip() for h in raw_header_line.strip().split(",")]
        cleaned_headers = [h.strip().strip('"').strip("'").lower() for h in raw_headers]

        # Use cleaned headers as fieldnames for DictReader
        reader = csv.DictReader(f, fieldnames=cleaned_headers)

        # For 2018 and earlier, the NPS unit name is located in RegionDescription. After
        # 2018, it's in ParentLocation.
        if int(file_path.name[:4]) < 2019:
            if reader.fieldnames is None:
                raise ValueError("Missing fieldnames — cleaned_headers is None")

            names = list(reader.fieldnames)
            i, j = names.index("regiondescription"), names.index("parentlocation")
            names[i], names[j] = names[j], names[i]
            reader.fieldnames = names

        # Prepare insert SQL (parameterized)
        PLACEHOLDERS = ", ".join(["?"] * len(expected_columns))
        INSERT_SQL = (
            f"INSERT INTO reservations ({', '.join(expected_columns)}) "
            f"VALUES ({PLACEHOLDERS})"
        )
        batch = []
        BATCH_SIZE = 10000

        # Begin one large transaction
        conn.execute("BEGIN")

        for row in reader:
            row["startdate"] = clean_date(row.get("startdate", ""))
            row["enddate"] = clean_date(row.get("enddate", ""))
            row["orderdate"] = clean_date(row.get("orderdate", ""))
            val = row.get("numberofpeople", "")
            row["numberofpeople"] = None if val == "" else val
            # row keys are cleaned_headers; build values in expected order
            values = [row.get(col, None) for col in expected_columns]
            batch.append(values)

            if len(batch) >= BATCH_SIZE:
                print(f"{file_path} - {BATCH_COUNT/lines * 10000 * 100:.1f}%")
                BATCH_COUNT += 1
                cur.executemany(INSERT_SQL, batch)
                batch = []

        # Final leftover (remainder) batch
        if batch:
            print("final batch")
            cur.executemany(INSERT_SQL, batch)

        conn.commit()

conn.close()

# %%
################
# Create indices
################
conn = sqlite3.connect("../data/reservations/reservations.db")
cur = conn.cursor()

cur.execute(
    "CREATE INDEX IF NOT EXISTS idx_parentlocation ON reservations(parentlocation);"
)

conn.commit()
conn.close()

# %%

import pandas as pd

conn = sqlite3.connect("../data/reservations/reservations.db")
cur = conn.cursor()

SQL_QUERY = """
            SELECT parentlocation, park, facilityid, startdate, enddate, orderdate, numberofpeople
            FROM reservations
            WHERE parentlocation = ?;
             """

df = pd.read_sql_query(SQL_QUERY, conn, params=("Joshua Tree National Park",))
conn.close()

# %%

fdsa = pd.read_csv("../data/reservations/2022.csv")
# %%
