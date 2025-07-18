"""
Model Code
"""

# %%

import datetime
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# %%
########################
# Import visitation data
########################
LEADTIME = datetime.timedelta(31)
PARK_CODE = "JOTR"
PARK_NAME = "Joshua Tree National Park"
# PARK_CODE = "GRSM"
# PARK_NAME = "Great Smoky Mountains National Park"
# PARK_CODE = "ZION"
# PARK_NAME = "Zion National Park"
# PARK_CODE = "DENA"
# PARK_NAME = "Denali National Park & Preserve"
# PARK_CODE = "YOSE"
# PARK_NAME = "Yosemite National Park"
# PARK_CODE = "YELL"
# PARK_NAME = "Yellowstone National Park"
# PARK_CODE = "GLAC"
# PARK_NAME = "Glacier National Park"

visits = pd.read_csv("../data/nps_visits.csv").astype({"date": "datetime64[s]"})
visits_jt = visits.loc[visits["park_code"] == PARK_CODE].set_index("date")["visits"]

# %%
#########################
# Import reservation data
#########################
conn = sqlite3.connect("../data/reservations/reservations.db")

SQL_QUERY = (
    """
            SELECT parentlocation, park, facilityid,
            SUBSTR(startdate, 1, 10) AS startdate,
            SUBSTR(enddate, 1, 10) AS enddate,
            SUBSTR(orderdate, 1, 10) AS orderdate,
            CASE
                WHEN numberofpeople = '' THEN NULL
                ELSE numberofpeople
            END AS numberofpeople
            FROM reservations
            WHERE parentlocation = '"""
    + PARK_NAME
    + """'
            AND orderdate LIKE '____-__-__%'
            AND startdate LIKE '____-__-__%'
            AND enddate LIKE '____-__-__%'
            AND orderdate NOT LIKE '0%'
            AND startdate NOT LIKE '0%'
            AND enddate NOT LIKE '0%';
             """
)
df = pd.read_sql_query(SQL_QUERY, conn)
conn.close()

# %%
############
# Clean data
############
df["facilityid"] = 12
# For number of people, replace nulls with mode
mode_value = df["numberofpeople"].mode().iloc[0]
df["numberofpeople"] = df["numberofpeople"].fillna(mode_value).astype(int)

# Convert strings to datetime
date_cols = ["startdate", "enddate", "orderdate"]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

# Drop reservations where start date is before order date
df = df.loc[~(df["startdate"] < df["orderdate"])]
# Drop reservations where start date is after end date
df = df.loc[~(df["startdate"] > df["enddate"])]

# Cast facilityid as category
df = df.astype({"facilityid": "category"})

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Manually editing here - suspicious that:
# - all cottonwood reservations should be considered the same
# - all Indian Cove reservations should be considered the same
# - Ryan campground is part of sheep pass group (less sure)
# - Backcountry permits are combined with JTNP Tours facility (less sure)
# - likely not true: Jumbo rocks with something else
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# df["facilityid"] = df["facilityid"].replace({272299: 232471})
# df["facilityid"] = df["facilityid"].replace({10053779: 232472})
# df["facilityid"] = df["facilityid"].replace({10056207: 232470})
# df["facilityid"] = df["facilityid"].replace({4675329: 300004})
# df["facilityid"] = df["facilityid"].replace({272300: 232472})

# %%
#######################################
# Plot visits and reservations together
#######################################

# Reservations by facility id and year
orders = (
    df.groupby([df["orderdate"].dt.year, df["facilityid"]])["numberofpeople"]
    .sum()
    .reset_index()
)
orders = orders.loc[(orders["orderdate"] > 2006) & (orders["orderdate"] < 2024)]

# visits by year
df_jt = visits.loc[
    (visits["park_code"] == PARK_CODE)
    & (visits["date"].dt.year > 2006)
    & (visits["date"].dt.year < 2024)
]
df_yearly = df_jt.groupby(df_jt["date"].dt.year)["visits"].sum()
fig_visits = px.line(
    data_frame=df_yearly, x=df_yearly.index, y="visits", labels={"value1": "Line Value"}
)

# Create a new figure and add traces from both
fig = go.Figure()

for facid in orders["facilityid"].unique():
    temp_df = orders.loc[orders["facilityid"] == facid]
    fig.add_trace(
        go.Bar(
            x=temp_df["orderdate"], y=temp_df["numberofpeople"], name=facid, yaxis="y"
        )
    )

fig.add_trace(
    go.Scatter(
        x=df_yearly.index,
        y=df_yearly,
        name="Visits",
        yaxis="y2",
        mode="lines+markers",
    )
)

# Optional layout tweaks
fig.update_layout(
    title="Combined Line and Bar Chart",
    yaxis=dict(
        title="Reservations",
        showgrid=False,
    ),
    yaxis2=dict(
        title="Visits", overlaying="y", side="right", rangemode="tozero", showgrid=False
    ),
    barmode="stack",
    template="plotly_white",
)

fig.show()

# %%
######################
# Prep for forecasting
######################

beginning = df["startdate"].min()
min_year = beginning.year
end = df["startdate"].max()
max_year = end.year

# Visits for just whatever park we're interested in
visits_jt = visits.loc[visits["park_code"] == PARK_CODE].set_index("date")["visits"]

# Weights for facilities that appear after data begin
df["month"] = df["startdate"].dt.month
df["year"] = df["startdate"].dt.year

all_months = sorted(df["month"].unique())
all_years = sorted(df["year"].unique())
all_facilities = sorted(df["facilityid"].unique())
fac_weights = pd.DataFrame(
    [(m, y, f) for m in all_months for y in all_years for f in all_facilities],
    columns=["month", "year", "facilityid"],
)
fac_weights["weight"] = pd.NA
facility_monthyears = df.groupby(["facilityid", "month"], observed=False)["year"].apply(
    set
)

# Calculate percentage for each month-year-facility
for month_ in all_months:
    for year in all_years:
        years_up_to_y = [yr for yr in all_years if yr <= year]
        for f in all_facilities:
            years_active = facility_monthyears.get((f, month_))

            if isinstance(years_active, (list, set, tuple)):
                NUM_YEARS_PRESENT = len(
                    [yr for yr in years_up_to_y if yr in years_active]
                )
            else:
                NUM_YEARS_PRESENT = 0

            mask = (
                (fac_weights["month"] == month_)
                & (fac_weights["year"] == year)
                & (fac_weights["facilityid"] == f)
            )
            fac_weights.loc[mask, "weight"] = pow(
                NUM_YEARS_PRESENT / len(years_up_to_y), 1
            )
df = df.merge(fac_weights, on=["month", "year", "facilityid"], how="left")
df["numberofpeopleweighted"] = df["numberofpeople"] * df["weight"]

####################
# Reservation matrix
####################
df["startmonthyear"] = df["startdate"].dt.to_period("M").dt.to_timestamp()

# This is not the exact lead time to the reservation but the lead time to the
# beginning of the month of the reservation which is what we need for forecasting
# monthly visitation before that month occurs.
df["monthleadtime"] = df["startmonthyear"] - df["orderdate"]

# This is the sum of reservations for each startdate with minimum monthleadtime.
res_matrix = (
    df.loc[df["monthleadtime"] >= LEADTIME]
    .groupby("startdate")["numberofpeople"]
    .sum()
    .reset_index()
)

# Aggregate to monthyears
res_matrix["startmonthyear"] = (
    res_matrix["startdate"].dt.to_period("M").dt.to_timestamp()
)
res_matrix = res_matrix.groupby("startmonthyear")["numberofpeople"].sum()
# Fill out the entire daterange index
full_index = pd.date_range(
    start=f"{min_year}-01-01", end=f"{max_year}-12-01", freq="MS"
)
res_matrix = res_matrix.reindex(full_index, fill_value=0)

###############
# Pickup matrix
###############
def pickup(y: int, month_of_year: int) -> float:
    """
    Calculate pickup given year, month, and leadtime.
    """
    reservation_count = res_matrix[datetime.datetime(y, month_of_year, 1)]
    reservation_count = reservation_count.mean()

    if reservation_count < 100:
        return np.nan

    return visits_jt[datetime.datetime(y, month_of_year, 1)] / reservation_count

pickup_matrix = {}
for year in range(min_year, max_year):
    for month in range(1, 13):
        pickup_matrix[(year, month, LEADTIME)] = pickup(year, month)

# %%
############################################################
# Forecast model for month=t with lead time=l
############################################################


def v_forecast(monthyear: datetime.datetime, leadtime: datetime.timedelta) -> float:
    """Multiplicative, (classical/advanced?), historical average pickup model.
    leadtime is timedelta in days.

    Will only generate a forecast if the month of interest is at least one year after
    the beginning of the relevant data.
    Will also only generate a forecast if the month of interest is at least the
    leadtime after the beginning.
    """
    if (monthyear > beginning + datetime.timedelta(366)) & (
        monthyear - leadtime > beginning
    ):
        return pickup_est(monthyear, leadtime) * tsa_res_count(
            monthyear.year, monthyear.month
        )
    return 0


def pickup_est(monthyear: datetime.datetime, leadtime: datetime.timedelta) -> float:
    """Estimator of pickup. Currently using a simple exponential smoothing model
    to generate estimator. This weights recent pickups more heavily
    """
    yearly_pickups = pd.Series(
        {
            datetime.datetime(y, monthyear.month, 1): pickup_matrix[
                (y, monthyear.month, leadtime)
            ]
            for y in range(min_year, monthyear.year)
        }
    )

    # If there are never any valid previous pickup values, then return nan
    if yearly_pickups.empty or yearly_pickups.isna().all():
        return np.nan

    # If only one usable value, return it
    non_na_values = yearly_pickups.dropna()
    if len(non_na_values) == 1 or non_na_values.nunique() == 1:
        return float(non_na_values.iloc[0])

    # Fill missing values with the mean (or consider ffill/bfill if trend exists)
    yearly_pickups = yearly_pickups.fillna(non_na_values.mean())

    yearly_pickups.index = pd.DatetimeIndex(
        pd.to_datetime(yearly_pickups.index)
    ).to_period("Y")
    yearly_pickups = yearly_pickups.sort_index()

    model = SimpleExpSmoothing(yearly_pickups).fit(smoothing_level=0.7, optimized=False)
    return model.forecast(1).iloc[0]


def tsa_res_count(y: int, month_of_year: int) -> float:
    """
    Weighted reservation counts then use a exponential smoothing function to forecast
    what "should" be the next reservation count.
    """
    res_trend = pd.Series(
        {
            datetime.datetime(year, month_of_year, 1): res_matrix[
                datetime.datetime(year, month_of_year, 1)
            ]
            for year in range(min_year, y)
        }
    )

    # If there is only one year of history, then return that one pickup
    if len(res_trend) == 1:
        return float(res_trend.iloc[0])
    # If there are never any valid previous pickup values, then return nan
    if res_trend.isna().all():
        return np.nan
    # If no trend or level, then return 0 to stop below from complaining.
    if (res_trend == 0).all():
        return 0

    res_trend.index = pd.DatetimeIndex(pd.to_datetime(res_trend.index)).to_period("Y")
    res_trend = res_trend.sort_index()
    model = SimpleExpSmoothing(res_trend).fit(smoothing_level=0.8, optimized=False)
    return model.forecast(1).iloc[0]


# %%
visits_df = visits_jt.loc[(pd.to_datetime(visits_jt.index).year >= 2008)].reset_index()
visits_df["vhat"] = visits_df["date"].apply(lambda date: v_forecast(date, LEADTIME))

visits_df = visits_df.melt(id_vars="date", value_vars=["visits", "vhat"])

px.line(visits_df.sort_values("date"), x="date", y="value", color="variable")

# %%
######
# Plots to look at what is happening under the hood of the model
######
res_diag = visits_jt.loc[(pd.to_datetime(visits_jt.index).year >= 2006)].reset_index()

res_diag["tsa_res"] = res_diag["date"].apply(
    lambda date: tsa_res_count(date.year, date.month)
)
res_diag["pickup_est"] = res_diag["date"].apply(
    lambda date: pickup_est(date, datetime.timedelta(31))
)
px.line(res_diag.sort_values("date"), x="date", y="tsa_res").show()
px.line(res_matrix).show()
px.line(res_diag.sort_values("date"), x="date", y="pickup_est").show()

# %%
# Looking at reservations by month

month_lead = df.loc[
    (df["startdate"].dt.month == 3)
    & (
        df["startdate"].dt.to_period("M").dt.to_timestamp() - df["orderdate"]
        >= datetime.timedelta(31)
    )
]
month_lead["year"] = month_lead["startdate"].dt.year
month_lead = (
    month_lead.groupby(["year", "facilityid"])["numberofpeople"].sum().reset_index()
)

px.line(
    month_lead.sort_values("year"),
    x="year",
    y="numberofpeople",
    color="facilityid",
)

# %%
