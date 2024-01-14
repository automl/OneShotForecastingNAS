# All the files in this package is originated from
# https://github.com/cure-lab/LTSF-Linear

from pathlib import Path
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
import warnings

SMALL_DATASET = ['ETTh1', 'ETTh2', 'illness', 'exchangerate']
LARGE_DATASET = ['ETTm1', 'ETTm2', 'traffic', 'weather']


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
    assert flag in ['train', 'val', 'test', 'train_val']
    type_map = {'train': 0, 'val': 1, 'test': 2}

    dataset_path = Path(root_path) / file_name

    scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(dataset_path))

    if dataset_name.startswith('ETTh'):
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    elif dataset_name.startswith('ETTm'):
        border1s = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0,         num_train,            len(df_raw) - num_test ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]


    border1 = 0
    if flag == 'train_val':
        border2 = border1s[2]
    elif flag == 'val':
        border2 = border1s[2]
    elif flag == 'train':
        border2 = border1s[1]
    elif flag == 'test':
        set_type = type_map[flag]
        border2 = border2s[set_type]
    else:
        raise NotImplementedError

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
    return df_raw, data, border1, border2, (border1s, border2s)


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
    df_raw, data, border1, border2, _ = get_ltsf_dataset(root_path, file_name=file_name,
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
            'X': None,
            'y':y_train,
            "start_times": start_times
        })
    elif series_type == 'S':
        Y = data[border1: border2].transpose()
        start_time = [pd.to_datetime(df_raw['date'][0]) for _ in range(len(Y))]
        dataset_info.update({
            'X': None,
            'y': Y,
            "start_times": start_time
        })
    return dataset_info, border2


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
                      train_only=False) -> tuple[dict, int, int, tuple]:
    df_raw, data, border1, border2, (border1s, border2s) = get_ltsf_dataset(root_path, file_name=file_name,
                                                      series_type=series_type, dataset_name=dataset_name, flag=flag,
                                                      target_name=target_name, do_normalization=do_normalization,
                                                      train_only=train_only)
    if flag != 'test':
        data = data[:border2]
    start_time = pd.to_datetime(df_raw['date'][0])
    if make_dataset_uni_variant:
        y = data.transpose()
        start_times = [start_time for _ in range(len(y))]
    else:
        y = [data]
        start_times = [start_time]

    dataset_info = {
        'X': None,
        'y': y,
        'start_times': start_times,
        'n_prediction_steps': forecasting_horizon,
        'freq': freq
    }

    return dataset_info, border2, len(data) - forecasting_horizon, (border1s, border2s)
    """
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
    """

