# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily

# The name of the column containing time series values after loading data from the .tsf file into a dataframe
VALUE_COL_NAME = "series_value"

# The name of the column containing timestamps after loading data from the .tsf file into a dataframe
TIME_COL_NAME = "start_timestamp"

SEASONALITY_MAP = {
    "minutely": [1440, 10080, 525960],
    "10_minutes": [144, 1008, 52596],
    "15_minutes": [96, 672, 35064],
    "half_hourly": [48, 336, 17532],
    "hourly": [24, 168, 8766],
    "daily": 7,
    "weekly": 365.25 / 7,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
    "minutely": "1min",
    "10_minutes": "10min",
    "15_minutes": "15min",
    "half_hourly": "30min",
    "hourly": "1H",
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1M",
    "quarterly": "1Q",
    "yearly": "1Y"
}
