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
PICKUP_SMOOTHING = .94 # .94 was great when tsa function didn't have +1
RES_SMOOTHING = .4 # .96
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
      SELECT parentlocation, park, facilityid, startdate, enddate, orderdate,
        numberofpeople
      FROM reservations
      WHERE parentlocation = ?
      AND orderdate NOT LIKE '0%'
      AND startdate NOT LIKE '0%'
      AND enddate NOT LIKE '0%';
    """
)
df = pd.read_sql_query(SQL_QUERY, conn, params=(PARK_NAME,))
conn.close()

# %%
############
# Clean data
############
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

# %%
#######################################
# Plot visits and reservations together
#######################################

# visits by year
df_jt = visits.loc[
    (visits["park_code"] == PARK_CODE)
    & (visits["date"].dt.year > 2006)
    & (visits["date"].dt.year < 2024)
]

# %%
######################
# Prep for forecasting
######################

beginning = df["startdate"].min()
min_year = beginning.year
end = df["startdate"].max()
max_year = end.year

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

    # Noise reduction. Get rid of values that are close enough to zero to be considered
    # noise. Somewhat arbitrary.
    if reservation_count < (res_matrix.max() * 0.01):
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
    """Multiplicative, classical, expsmooth pickup model.
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
    return np.nan


def pickup_est(monthyear: datetime.datetime, leadtime: datetime.timedelta) -> float:
    """Estimator of pickup. Currently using a simple exponential smoothing model
    to generate estimator. This weights recent pickups more heavily
    """

    pickup_return = 0

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
        pickup_return = float(non_na_values.iloc[0])

    else:
        # Fill missing values with the mean (or consider ffill/bfill if trend exists)
        yearly_pickups = yearly_pickups.fillna(non_na_values.mean())

        yearly_pickups.index = pd.DatetimeIndex(
            pd.to_datetime(yearly_pickups.index)
        ).to_period("Y")
        yearly_pickups = yearly_pickups.sort_index()

        # print(yearly_pickups)

        model = SimpleExpSmoothing(yearly_pickups).fit(
            smoothing_level=PICKUP_SMOOTHING, optimized=False
        )
        pickup_return =  model.forecast(1).iloc[0]

    # Arbitrary noise reduction.
    if pickup_return > 1000:
        return np.nan

    return pickup_return


def tsa_res_count(y: int, month_of_year: int) -> float:
    """
    Reservation counts then use a exponential smoothing function to forecast
    what "should" be the next reservation count.
    """
    res_trend = pd.Series(
        {
            datetime.datetime(year, month_of_year, 1): res_matrix[
                datetime.datetime(year, month_of_year, 1)
            ]
            for year in range(min_year, y + 1) # works better with + 0 though. IDK
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
    # print(res_trend)
    res_trend = res_trend.sort_index()
    model = SimpleExpSmoothing(res_trend).fit(
        smoothing_level=RES_SMOOTHING, optimized=False
    )
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
    lambda date: pickup_est(date, LEADTIME)
)
px.line(res_diag.sort_values("date"), x="date", y="tsa_res").show()
px.line(res_matrix).show()
px.line(res_diag.sort_values("date"), x="date", y="pickup_est").show()

# # %%
# # Looking at reservations by month

# month_lead = df.loc[
#     (df["startdate"].dt.month == 3)
#     & (
#         df["startdate"].dt.to_period("M").dt.to_timestamp() - df["orderdate"]
#         >= datetime.timedelta(31)
#     )
# ]
# month_lead["year"] = month_lead["startdate"].dt.year
# month_lead = (
#     month_lead.groupby(["year", "facilityid"])["numberofpeople"].sum().reset_index()
# )

# px.line(
#     month_lead.sort_values("year"),
#     x="year",
#     y="numberofpeople",
#     color="facilityid",
# )

# %%

####################
# Five year AR model
####################

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.statespace.sarimax import SARIMAX

LEADTIME_MONTHS = 1

visits = pd.read_csv("../data/nps_visits.csv").astype({"date": "datetime64[s]"})
visits_jt = visits.loc[visits["park_code"] == PARK_CODE].set_index("date")["visits"]

visits_jt.index = pd.to_datetime(visits_jt.index, errors="coerce")
visits_jt = visits_jt.asfreq("MS")
visits_jt = visits_jt.sort_index()

predictions = pd.Series(index=visits_jt.index, dtype=float)
sarima_preds = pd.Series(index=visits_jt.index, dtype=float)

# Iterate over all dates where we want to make a forecast
for date in visits_jt.index:
    target_month = date.month
    target_year = date.year

    # Subset of past years' same month
    mask = (visits_jt.index.month == target_month) & (
        visits_jt.index.year < target_year
    )
    subset = visits_jt.loc[mask].dropna()

    if len(subset) >= 12:
        lag_order = ar_select_order(subset, maxlag=5, ic='aic')
        model = AutoReg(subset, lags=lag_order.ar_lags, old_names=False)
        result = model.fit()
        predictions.loc[date] = result.forecast(1).iloc[0]
    else:
        # Not enough data to fit AR(5); optionally set prediction to NaN
        predictions.loc[date] = np.nan

predictions.name = "ar_model"
models_wide = visits_df.pivot(index="date", columns="variable", values="value")
models_wide = models_wide.merge(predictions, on="date")


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Mean Absolute Percentage Error
    """

    actual = actual.replace(0, None)

    return ((actual - forecast) / actual).abs().mean() * 100


def mse(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Mean Absolute Percentage Error
    """

    actual = actual.replace(0, None)

    return pow(actual - forecast, 2).mean()


def rsquared(actual: pd.Series, forecast: pd.Series) -> float:
    """
    R squared
    """
    ss_res = pow(actual - forecast, 2).sum()
    ss_tot = pow(actual - actual.mean(), 2).sum()
    return 1 - ss_res / ss_tot


print("pickup model")
print(f"MAPE= {mape(models_wide['visits'], models_wide['vhat']):.2f}")
print(f"MSE= {mse(models_wide['visits'], models_wide['vhat']):.2f}")
print(f"R^2= {rsquared(models_wide['visits'], models_wide['vhat']):.2f}")

print("AR model:")
print(f"MAPE= {mape(models_wide['visits'], models_wide['ar_model']):.2f}")
print(f"MSE= {mse(models_wide['visits'], models_wide['ar_model']):.2f}")
print(f"R^2= {rsquared(models_wide['visits'], models_wide['ar_model']):.2f}")

# %%
# Need to get rid of April through August of 2020 because of Covid