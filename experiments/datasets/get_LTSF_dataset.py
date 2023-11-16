# All the files in this package is originated from
# https://github.com/cure-lab/LTSF-Linear

from pathlib import Path
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
import warnings


DATASETS_INFO = {
    "electricity": ("electricity/electricity.csv", 'M', '1H', [96, 192, 336, 720]),
    "ETTh1": ("ETT-small/ETTh1.csv", 'M', '1H', [96, 192, 336, 720]),
    "ETTh2": ("ETT-small/ETTh2.csv", 'M', '1H', [96, 192, 336, 720]),
    "ETTm1": ("ETT-small/ETTm1.csv", 'M', '1min', [96, 192, 336, 720]),
    "ETTm2": ("ETT-small/ETTm2.csv", 'M', '1min', [96, 192, 336, 720]),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "M", "1D", [96, 192, 336, 720]),
    "illness": ("illness/national_illness.csv", "M", "1W", [24, 36, 48, 60]),
    "traffic": ("traffic/traffic.csv", "M", "1H", [96, 192, 336, 720]),
    "weather": ("weather/weather.csv", "M", "10min", [96, 192, 336, 720])
}


def get_ltsf_dataset(root_path,
                      file_name:str,
                      series_type: str='M',
                      dataset_name: str = 'ETT_H',
                      flag='train',
                      target_name: str = 'OT',
                      do_normalization: bool = True,
                      train_only=False):
    assert flag in ['train', 'test']
    type_map = {'train': 0, 'test': 1}
    set_type = type_map[flag]

    dataset_path = Path(root_path) / file_name

    scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(dataset_path))

    if dataset_name.startswith('ETTh'):
        border1s = [0,                          12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset_name.startswith('ETTm'):
        border1s = [0,                                  12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border2s = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * (0.7 if not train_only else 1))
        num_test = int(len(df_raw) * 0.2)

        border1s = [0, len(df_raw) - num_test ]
        border2s = [num_train, len(df_raw)]

    border1 = 0
    if flag == 'test':
        border2 = border1s[set_type]
    else:
        border2 = border1s[1]

    if dataset_name.startswith('ETT'):
        if series_type == 'M' or series_type == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif series_type == 'S':
            df_data = df_raw[[target_name]]
        else:
            raise NotImplementedError("Unknown Target Type!!!")
    else:
        cols = list(df_raw.columns)
        if series_type == 'S':
            cols.remove(target_name)
        cols.remove('date')

        if series_type == 'M' or series_type == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif series_type == 'S':
            df_raw = df_raw[['date'] + cols + [target_name]]
            df_data = df_raw[[target_name]]
        else:
            raise NotImplementedError("Unknown Target Type!!!")

    if do_normalization:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values
    return df_raw, data, border1, border2


def get_train_dataset(root_path,
                      file_name:str,
                      series_type: str='M',
                      freq: str = '1H',
                      forecasting_horizon: int = 96,
                      dataset_name: str = 'ETT_H',
                      flag='train',
                      target_name: str = 'OT',
                      do_normalization: bool = True,
                      make_dataset_uni_variant: bool = False,
                      train_only=False) -> dict:
    assert flag == 'train'
    df_raw, data, border1, border2 = get_ltsf_dataset(root_path, file_name=file_name,
                     series_type=series_type, dataset_name=dataset_name,flag=flag,
                     target_name=target_name, do_normalization=do_normalization,
                     train_only=train_only)
    dataset_info = {
        'n_prediction_steps': forecasting_horizon,
        'freq': freq
    }

    if series_type == 'M' or series_type == 'MS':
        # For multi-variant and multi-target datasets, we simply consider the entire dataset as one single
        # time series sequence
        start_time = pd.to_datetime(df_raw['date'][0])
        if make_dataset_uni_variant:
            y_train = data[border1: border2].transpose()
            start_times = [start_time for _ in range(len(y_train))]
        else:
            y_train = [data[border1: border2]]
            start_times = [start_time]
        dataset_info.update({
            'X_train': None,
            'y_train':y_train,
            "start_times": start_times
        })
    elif series_type == 'S':
        Y = data[border1: border2].transpose()
        start_time = [pd.to_datetime(df_raw['date'][0]) for _ in range(len(Y))]
        dataset_info.update({
            'X_train': None,
            'y_train': Y,
            "start_times": start_time
        })
    return dataset_info


def get_test_dataset(root_path,
                      file_name:str,
                      series_type: str='M',
                      freq: str = '1H',
                      forecasting_horizon: int = 96,
                      dataset_name: str = 'ETT_H',
                      flag='test',
                      target_name: str = 'OT',
                      do_normalization: bool = True,
                      make_dataset_uni_variant:bool = False,
                      train_only=False) -> dict:
    assert flag == 'test'
    df_raw, data, border1, border2 = get_ltsf_dataset(root_path, file_name=file_name,
                                                      series_type=series_type, dataset_name=dataset_name, flag=flag,
                                                      target_name=target_name, do_normalization=do_normalization,
                                                      train_only=train_only)
    for idx in range(border2, len(data) - forecasting_horizon):
        Y = data[: idx]
        start_time = pd.to_datetime(df_raw['date'][0])
        if make_dataset_uni_variant:
            Y = Y.transpose()
            start_times = [start_time for _ in range(len(Y))]
        else:
            Y = [Y]
            start_times = [start_time]
        dataset_info = {
            'X_test': None,
            'past_targets': Y,
            'future_targets': None,
            'start_times': start_times,
        }
        yield dataset_info

