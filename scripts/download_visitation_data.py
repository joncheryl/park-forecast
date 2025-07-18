"""
Download monthly visitation data for National Park dashboard.

No visitation data found for the following NPS units:
# No data because new NPS units:
["CEBE", "TILL"]
# Don't know why no data:
["BICR", "BLRV", "CAMO", "EBLA", "FOMR", "FRRI", "HART", "HONO", "NEPH", "SPRA"]

Deprecated NPS units:
["NACA", "NAVC"]
"""

# %%

import re
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup

# %%
# Obtain directory of parks
# There is a directory here, but it has errors:
# 'https://www.nps.gov/aboutus/foia/upload/NPS-Unit-List.xlsx'

DIRECTORY_URL = "https://irma.nps.gov/Stats/Reports/Park"
response = requests.get(DIRECTORY_URL, timeout=100)
html = response.text

# Search page for the dictionary containing the parks directory
PATTERN = r"unitData\s*:\s*(\[\{.*?\}\])"
match = re.search(PATTERN, html, re.DOTALL)

# If you find it, then format it from json string -> list -> dataframe ->
# cleaned dataframe
if match:
    unit_data_json = match.group(1)
    unit_data = json.loads(unit_data_json)

    directory = pd.DataFrame(unit_data)
    directory = directory.rename(columns={"Text": "park_name", "Value": "park_code"})
else:
    directory = pd.DataFrame({})
    print("Could not find unitData (the directory dictionary) in the page.")

# Get proper names for each unit by saving the 'title' of each respective NPS website.
websites = "https://www.nps.gov/" + directory["park_code"].str.lower() + "/index.htm"


def get_title(url):
    """Function obtaining proper park unit name from webpage title."""
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            title = "No title found"

        # Remove any final parenthetical at the end of the title
        title = re.sub(r"\s*\([^()]*\)\s*$", "", title)

        return title
    except requests.exceptions.ConnectionError as e:
        return f"Error: Could not connect to the URL. ({e})"
    except requests.exceptions.Timeout as e:
        return f"Error: The request timed out. ({e})"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error occurred: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: An unexpected error occurred during the request: {e}"


directory["park_name"] = [get_title(unit_page) for unit_page in websites]
directory["park_name"] = directory["park_name"].str.replace(" - ", "-")

#######################################################################################
#################################### MANUAL EDITS #####################################
#######################################################################################

corrections = {
    "PAAV": "Pennsylvania Avenue National Historic Site",
    "JOFK": "John F. Kennedy Memorial Center for the Performing Arts",
    "KICA": "Kings Canyon National Park",
    "NCPC": "National Mall and Memorial Parks",
    "NCPE": "National Capital Parks-East",
    "OBRI": "Obed Wild and Scenic River",
    "PRPA": "President's Park",
    "SEQU": "Sequoia National Park",
    "JEFM": "Thomas Jefferson Memorial",
    "JODR": "John D. Rockefeller Jr. Memorial Parkway", # NPS site incorrectly goes to Grand Teton NPS site #pylint: disable=line-too-long
    "LACH": "Lake Chelan National Recreation Area", # NPS site incorrectly goes to North Cascade NPS site #pylint: disable=line-too-long
    "ROLA": "Ross Lake National Recreation Area", # NPS site incorrectly goes to North Cascade NPS site #pylint: disable=line-too-long
    "FOCA": "Fort Caroline National Memorial", # NPS site incorrectly goes to North Cascade NPS site #pylint: disable=line-too-long
}

for park_code, park_name in corrections.items():
    directory.loc[directory["park_code"] == park_code, "park_name"] = park_name

# Remove NACA and NAVC from directory (deprecated).
directory = directory[~directory["park_code"].isin(["NACA", "NAVC"])]

# %%
# Download visitation data
URL_PREFIX = (
    "https://irma.nps.gov/Stats/MvcReportViewer.aspx?_id=2588c518-79e7-494a-b9"
    "88-1d126584c7d8&_m=Remote&_r=%2fNPS.Stats.Reports%2fPark+Specific+Reports"
    "%2fRecreation+Visitors+By+Month+(1979+-+Last+Calendar+Year)&_15=True&_16="
    "True&_18=True&_19=True&_34=False&_35=False&_39=880px&Park="
)

directory_dict = {}
unfound_parks = []

for park_abr in directory["park_code"].unique():
    try:
        directory_dict[park_abr] = pd.read_html(
            URL_PREFIX + park_abr,
            attrs={"cols": "14"},
            header=1,  # skiprows=[2] why do I have this here?
        )[0]
    except ValueError as e:
        print(f"Failed to fetch {park_abr}: {e}")
        unfound_parks.append(park_abr)

# Find empty dataframes and drop them from dictionary.
empties = [key for (key, value) in directory_dict.items() if value.shape[0] == 0]
for key in empties:
    print("Deleting " + key)
    del directory_dict[key]

# %%
# Combine all park visitation data together into one dataframe.
visits_df = pd.concat(directory_dict, names=["park_code"]).reset_index(
    level="park_code"
)

# Shape dataframe into format I want.
month_names = visits_df.columns[1:-1]
visits_df = visits_df.melt(
    id_vars=["park_code", "Year"],
    var_name="Month",
    value_vars=list(month_names),
    value_name="visits",
)

# Only keep 2024 and older data
visits_df = visits_df.loc[visits_df["Year"] < 2025]

# Convert seperate year and month columns into one datetime column
visits_df["date"] = pd.to_datetime(
    visits_df["Year"].astype(str) + " " + visits_df["Month"].str.title(), format="%Y %b"
)
visits_df = visits_df.drop(columns=["Year", "Month"])

# Tack on proper unit names for wikipedia-ing.
visits_df = visits_df.merge(directory, on="park_code", how="outer")

# %%
###############################################################################
############################# Write data to file. #############################
###############################################################################
visits_df.to_csv("../data/nps_visits.csv", index=False)
