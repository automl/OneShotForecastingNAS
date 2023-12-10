import numpy as np
import torch
from functools import partial

from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from sklearn.model_selection import TimeSeriesSplit
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.base_dataset import TransformSubset
from autoPyTorch.utils.common import custom_collate_fn

from autoPyTorch.pipeline.components.training.data_loader.time_series_util import PadSequenceCollector


def get_forecasting_dataset(n_prediction_steps,
                            freq,
                            X_train,
                            y_train,
                            start_times,
                            dataset_name='forecasting_dataset'):
    # Create a validator object to make sure that the data provided by
    # the user matches the autopytorch requirements
    input_validator = TimeSeriesForecastingInputValidator(
        is_classification=False,
    )

    # Fit an input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    input_validator.fit(
        X_train=X_train,
        y_train=y_train,
        start_times=start_times,
    )

    dataset = TimeSeriesForecastingDataset(
        X=X_train,
        Y=y_train,
        dataset_name=dataset_name,
        freq=freq,
        start_times=start_times,
        validator=input_validator,
        n_prediction_steps=n_prediction_steps,
    )
    dataset.transform_time_features = True
    return dataset


def regenerate_splits(dataset: TimeSeriesForecastingDataset, val_share: float, start_idx: int | None = None,
                      n_folds: int = 5):
    # AutoPyTorch dataset does not allow overlap between different validation series (such that each time step will be
    # assigned the same weights). Therefore, here we reimplement train / validation splits
    strategy = 'holdout'

    if strategy == 'holdout':
        n_prediction_steps = dataset.n_prediction_steps
        splits = [[() for _ in range(len(dataset.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, seq in enumerate(dataset.datasets):
            if start_idx is not None:
                assert start_idx < len(seq), f"start_idx should be smaller than the length of the sequence." \
                                               f" however, they are {start_idx} and {len(seq)} respectively"
                indices = np.arange(start_idx, len(seq)) + idx_start
            else:
                indices = np.arange(len(seq)) + idx_start

            cv = TimeSeriesSplit(n_splits=2,
                                 test_size=int(val_share * (len(indices)-n_prediction_steps)),
                                 gap=n_prediction_steps - 1)

            train, val = list(cv.split(indices))[-1]
            splits[0][idx_seq] = indices[train]
            splits[1][idx_seq] = indices[val]
            idx_start += dataset.sequence_lengths_train[idx_seq]
        train_indices_dataset = np.hstack([sp for sp in splits[0]])
        val_indices_dataset = np.hstack([sp for sp in splits[1]])
        return (train_indices_dataset, val_indices_dataset)
    elif strategy == 'cv':

        splits = [[() for _ in range(len(dataset.datasets))] for _ in range(2)]
        idx_start = 0
        n_prediction_steps = dataset.n_prediction_steps
        for idx_seq, seq in enumerate(dataset.datasets):
            if start_idx is not None:
                assert start_idx < len(seq), f"start_idx should be smaller than the length of the sequence." \
                                               f" however, they are {start_idx} and {len(seq)} respectively"
                indices = np.arange(start_idx, len(seq)) + idx_start
            else:
                indices = np.arange(len(seq)) + idx_start

            cv = TimeSeriesSplit(n_splits=2,
                                 test_size=int(val_share * (len(indices) - n_prediction_steps)),
                                 gap=n_prediction_steps - 1)
            train, val = list(cv.split(indices))[-1]

            n_each_split = (len(indices) - n_prediction_steps) / (2 * n_folds)

            splits[0][idx_seq] = np.concatenate([indices[int(2 * i* n_each_split): int(2* i* n_each_split + n_each_split)]
                                                 for i in range(n_folds)])
            splits[1][idx_seq] = np.concatenate([indices[int((2 * i + 1) * n_each_split):
                                                         int((2* i + 1) * n_each_split + n_each_split)] for i in range(n_folds)])

        train_indices_dataset = np.hstack([sp for sp in splits[0]])
        val_indices_dataset = np.hstack([sp for sp in splits[1]])
        return (train_indices_dataset, val_indices_dataset)
    else:
        raise NotImplementedError


def get_dataloader(dataset: TimeSeriesForecastingDataset,
                   splits: tuple[np.ndarray, np.ndarray],
                   batch_size: int,
                   window_size: int,
                   num_workers: int = 8,
                   padding_value: float = 0.0
                   ):
    n_prediction_steps = dataset.n_prediction_steps

    max_lagged_value = max(dataset.lagged_value)
    max_lagged_value += window_size + n_prediction_steps

    padding_collector = PadSequenceCollector(window_size=window_size,
                                             sample_interval_red_seq_len=1,
                                             sample_interval_fix_seq_len=1,
                                             target_padding_value=padding_value,
                                             seq_max_length=max_lagged_value)

    train_dataset = TransformSubset(dataset=dataset, indices=splits[0], train=True)
    val_dataset = TransformSubset(dataset=dataset, indices=splits[1], train=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, x_collector=padding_collector),
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, x_collector=padding_collector),
    )
    return train_data_loader, val_data_loader
