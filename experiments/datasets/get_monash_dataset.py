import pandas as pds
from datetime import datetime
import warnings
import os
import copy
from pathlib import Path
from .constant import VALUE_COL_NAME, TIME_COL_NAME, SEASONALITY_MAP, FREQUENCY_MAP, DATASETS

from datetime import datetime
from numpy import distutils
import distutils
import pandas as pd
import numpy as np


def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with = 'NaN', value_column_name = "series_value"):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"): # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (len(line_content) != 3):  # Attributes have both name and type
                                raise ValueError("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise ValueError("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(distutils.util.strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(distutils.util.strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise ValueError("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise ValueError("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise ValueError("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise ValueError("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if(len(series) == 0):
                            raise ValueError("A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if (numeric_series.count(replace_missing_vals_with) == len(numeric_series)):
                            raise ValueError("All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                            else:
                                raise ValueError("Invalid attribute type.") # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if(att_val == None):
                                raise ValueError("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise ValueError("Empty file.")
        if len(col_names) == 0:
            raise ValueError("Missing attribute section.")
        if not found_data_section:
            raise ValueError("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length


def get_train_dataset(root_path,
                      dataset_name: str = 'm4_hourly',
                      file_name: str = 'm4_hourly_dataset.tsv',
                      external_forecast_horizon: int | None = None,
                      for_validation=False,
                      ) -> dict:

    dataset_path = Path(root_path) / file_name

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = \
        convert_tsf_to_dataframe(str(dataset_path))

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required prediction steps")
        else:
            forecast_horizon = external_forecast_horizon

    train_series_list = []
    test_series_list = []

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1


    shortest_sequence = np.inf
    train_start_time_list = []

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime('1900-01-01 00-00-00',
                                                 '%Y-%m-%d %H-%M-%S')  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False
        train_start_time_list.append(pds.Timestamp(train_start_time))

        series_data = row[VALUE_COL_NAME].to_numpy()
        # Creating training and test series. Test series will be only used during evaluation
        if not for_validation:
            train_series_data = series_data[:len(series_data) - forecast_horizon]
            test_series_data = series_data[(len(series_data) - forecast_horizon): len(series_data)]

            y_test.append(series_data[-forecast_horizon:])
        else:
            train_series_data = series_data[:len(series_data) - 2 * forecast_horizon]
            test_series_data = series_data[(len(series_data) - 2 * forecast_horizon): len(series_data) - forecast_horizon]

            y_test.append(series_data[-2 * forecast_horizon:-forecast_horizon])

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        shortest_sequence = min(len(train_series_data), shortest_sequence)
    """
    if validation == 'cv':
        n_splits = 3
        while shortest_sequence - forecast_horizon - forecast_horizon * n_splits <= 0:
            n_splits -= 1

        if n_splits >= 2:
            resampling_strategy = CrossValTypes.time_series_cross_validation
            resampling_strategy_args = {'num_splits': n_splits}

        else:
            warnings.warn('The dataset is not suitable for cross validation, we will try holdout instead')
            validation = 'holdout'
    elif validation == 'holdout_ts':
        resampling_strategy = CrossValTypes.time_series_ts_cross_validation
        resampling_strategy_args = None
    if validation == 'holdout':
        resampling_strategy = HoldoutValTypes.time_series_hold_out_validation
        resampling_strategy_args = None
    """


    X_train = copy.deepcopy(train_series_list)
    y_train = copy.deepcopy(train_series_list)

    dataset_info = {
        'n_prediction_steps': forecast_horizon,
        'freq': freq,
        'X': None,
        'y': y_train,
        'start_times': train_start_time_list
    }
    return dataset_info, y_test


