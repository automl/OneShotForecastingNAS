# All the files in this package is originated from
# https://github.com/cure-lab/LTSF-Linear

from pathlib import Path
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

LARGE_DATASET = ['pems03', 'pems04', 'pems07', 'pems08']


def get_pems_dataset(root_path,
                     file_name: str,
                     series_type: str = 'M',
                     dataset_name: str = 'pems03',
                     flag='train',
                     window_size:int = 0,
                     target_name: str = 'OT',
                     do_normalization: bool = True,
                     train_only=False):
    assert flag in ['train', 'val', 'test', 'train_val']
    type_map = {'train': 0, 'val': 1, 'test': 2}

    dataset_path = Path(root_path) / file_name

    scaler = StandardScaler()

    data_raw = np.load(os.path.join(dataset_path), allow_pickle=True)
    lst = data_raw.files

    data_raw = data_raw['data'][:, :, 0]

    num_train = int(len(data_raw) * 0.6)
    num_vali = int(len(data_raw) * 0.8) - num_train
    num_test = len(data_raw) - num_train - num_vali

    border1s = [0, num_train + window_size, len(data_raw) - num_test + window_size]
    border2s = [num_train, num_train + num_vali, len(data_raw)]

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

    if do_normalization:
        train_data = data_raw[border1s[0]:border2s[0]]
        scaler.fit(train_data)
        data = scaler.transform(data_raw)
    else:
        data = data_raw
    total_data = np.split(data, [num_train, num_train + num_vali])

    for i in range(len(total_data)):
        df = pd.DataFrame(total_data[i])
        total_data[i] = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
    # fill the values
    data = np.concatenate(total_data)

    return data_raw, data, border1, border2, (border1s, border2s)


def get_train_dataset(root_path,
                      file_name: str,
                      series_type: str = 'M',
                      freq: str = '1H',
                      forecasting_horizon: int = 96,
                      dataset_name: str = 'ETT_H',
                      flag='train',
                      target_name: str = 'OT',
                      do_normalization: bool = True,
                      make_dataset_uni_variant: bool = False,
                      train_only=False) -> dict:
    df_raw, data, border1, border2, _ = get_pems_dataset(root_path, file_name=file_name,
                                                         series_type=series_type, dataset_name=dataset_name, flag=flag,
                                                         target_name=target_name, do_normalization=do_normalization,
                                                         train_only=train_only)
    dataset_info = {
        'n_prediction_steps': forecasting_horizon,
        'freq': freq
    }

    if series_type == 'M' or series_type == 'MS':
        # For multi-variant and multi-target datasets, we simply consider the entire dataset as one single
        # time series sequence
        start_time = None
        if make_dataset_uni_variant:
            y_train = data[border1: border2].transpose()
            start_times = [start_time for _ in range(len(y_train))]
        else:
            y_train = [data[border1: border2]]
            start_times = [start_time]
        dataset_info.update({
            'X': None,
            'y': y_train,
            "start_times": None
        })
    elif series_type == 'S':
        Y = data[border1: border2].transpose()
        start_time = [None for _ in range(len(Y))]
        dataset_info.update({
            'X': None,
            'y': Y,
            "start_times": None
        })
    return dataset_info, border2


def get_test_dataset(root_path,
                     file_name: str,
                     series_type: str = 'M',
                     freq: str = '1H',
                     forecasting_horizon: int = 96,
                     window_size: int = 0,
                     dataset_name: str = 'ETT_H',
                     flag='test',
                     target_name: str = 'OT',
                     do_normalization: bool = True,
                     make_dataset_uni_variant: bool = False,
                     train_only=False) -> tuple[dict, int, int, tuple]:
    df_raw, data, border1, border2, (border1s, border2s) = get_pems_dataset(root_path, file_name=file_name,
                                                                            series_type=series_type,
                                                                            dataset_name=dataset_name, flag=flag,
                                                                            target_name=target_name,
                                                                            do_normalization=do_normalization,
                                                                            window_size=window_size,
                                                                            train_only=train_only)
    if flag != 'test':
        data = data[:border2]
    if make_dataset_uni_variant:
        y = data.transpose()
    else:
        y = [data]

    dataset_info = {
        'X': None,
        'y': y,
        'start_times': None,
        'n_prediction_steps': forecasting_horizon,
        'freq': freq
    }

    return dataset_info, border2, len(data) - forecasting_horizon, (border1s, border2s)
