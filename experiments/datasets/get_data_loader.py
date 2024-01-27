import numpy as np
import torch
from functools import partial

from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from sklearn.model_selection import TimeSeriesSplit
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.base_dataset import TransformSubset
from autoPyTorch.utils.common import custom_collate_fn

from autoPyTorch.pipeline.components.training.data_loader.time_series_util import PadSequenceCollector, \
    TimeSeriesSampler


def get_forecasting_dataset(n_prediction_steps,
                            freq,
                            X,
                            y,
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
        X_train=X,
        y_train=y,
        start_times=start_times,
    )

    dataset = TimeSeriesForecastingDataset(
        X=X,
        Y=y,
        dataset_name=dataset_name,
        freq=freq,
        start_times=start_times,
        validator=input_validator,
        n_prediction_steps=n_prediction_steps,
    )
    dataset.transform_time_features = True
    return dataset


def regenerate_splits(dataset: TimeSeriesForecastingDataset,
                      val_share: float,
                      start_idx: int | None = None,
                      splits_ms: list[np.ndarray] | None = None,
                      n_folds: int = 5, strategy: str = 'cv'):
    # AutoPyTorch dataset does not allow overlap between different validation series (such that each time step will be
    # assigned the same weights). Therefore, here we reimplement train / validation splits
    if splits_ms is not None:
        # This applies the same split to each of the series. It is especially useful if we would like to
        splits = [[() for _ in range(len(dataset.datasets))] for _ in range(len(splits_ms))]
        idx_start = 0
        for idx_seq, seq in enumerate(dataset.datasets):
            for i, split_ms in enumerate(splits_ms):
                assert split_ms.max() < len(seq)

                splits[i][idx_seq] = split_ms + idx_start
            idx_start += dataset.sequence_lengths_train[idx_seq]
        all_splits = [np.hstack([sp for sp in split]) for split in splits]
        return all_splits
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
                                 test_size=int(val_share * (len(indices) - n_prediction_steps)),
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

            splits[0][idx_seq] = np.concatenate(
                [indices[int(2 * i * n_each_split): int(2 * i * n_each_split + n_each_split)]
                 for i in range(n_folds)])
            splits[1][idx_seq] = np.concatenate([indices[int((2 * i + 1) * n_each_split):
                                                         int((2 * i + 1) * n_each_split + n_each_split)] for i in
                                                 range(n_folds)])

        train_indices_dataset = np.hstack([sp for sp in splits[0]])
        val_indices_dataset = np.hstack([sp for sp in splits[1]])
        return (train_indices_dataset, val_indices_dataset)
    else:
        raise NotImplementedError


def get_dataset_sampler(dataset: TimeSeriesForecastingDataset,
                        split: np.ndarray,
                        batch_size: int,
                        num_batches_per_epoch: int,
                        sample_strategy='SeqUniform'
                        ):
    num_instances_all = np.size(split)
    num_instances_loader = num_batches_per_epoch * batch_size

    dataset_seq_length_train_all = dataset.sequence_lengths_train
    if np.sum(dataset_seq_length_train_all) == len(split):
        # this works if we want to fit the entire datasets
        seq_length = np.array(dataset_seq_length_train_all)
    else:
        _, seq_length = np.unique(split - np.arange(len(split)), return_counts=True)
    # create masks for masking
    seq_idx_inactivate = np.where(np.random.rand(seq_length.size) > 1.0)[0]
    if len(seq_idx_inactivate) == seq_length.size:
        seq_idx_inactivate = np.random.choice(seq_idx_inactivate, len(seq_idx_inactivate) - 1, replace=False)
    # this budget will reduce the number of samples inside each sequence, e.g., the samples becomes more sparse
    min_start = 0
    """
    num_instances_per_seqs = np.ceil(
        np.ceil(num_instances_train / (num_instances_dataset - min_start) * seq_train_length) *
        fraction_samples_per_seq
    )
    """
    if sample_strategy == 'LengthUniform':
        available_seq_length = seq_length - min_start
        available_seq_length = np.where(available_seq_length <= 1, 1, available_seq_length)
        num_instances_per_seqs = num_instances_loader / num_instances_all * available_seq_length
    elif sample_strategy == 'SeqUniform':
        num_seq_train = len(seq_length)
        num_instances_per_seqs = np.repeat(num_instances_loader / num_seq_train, num_seq_train)
    else:
        raise NotImplementedError(f'Unsupported sample strategy: {sample_strategy}')

    num_instances_per_seqs[seq_idx_inactivate] = 0

    sampler_indices = np.arange(num_instances_all)

    sampler = TimeSeriesSampler(indices=sampler_indices, seq_lengths=seq_length,
                                num_instances_per_seqs=num_instances_per_seqs,
                                min_start=min_start)
    return sampler


def get_dataloader(dataset: TimeSeriesForecastingDataset,
                   splits: tuple[np.ndarray],
                   batch_size: int,
                   window_size: int,
                   num_workers: int = 2,
                   num_batches_per_epoch: int | None = None,
                   is_test_sets: list[int] | None = None,
                   padding_value: float = 0.0,
                   batch_size_test: int | None = None,
                   sample_interval: int = 1,
                   ):
    n_prediction_steps = dataset.n_prediction_steps

    max_lagged_value: int = max(dataset.lagged_value)
    max_lagged_value += window_size + n_prediction_steps

    padding_collector = PadSequenceCollector(window_size=window_size,
                                             sample_interval_red_seq_len=sample_interval,
                                             sample_interval_fix_seq_len=1,
                                             target_padding_value=padding_value,
                                             seq_max_length=max_lagged_value)

    all_loader = []
    if is_test_sets is None:
        is_test_sets = [False] * len(splits)
    else:
        assert len(is_test_sets) == len(splits)
    for is_test, split in zip(is_test_sets, splits):
        sub_dataset = TransformSubset(dataset=dataset, indices=split, train=True)
        if num_batches_per_epoch is not None and len(split) < num_batches_per_epoch * batch_size:
            num_batches_per_epoch = None
        if num_batches_per_epoch is None or is_test:
            if batch_size_test is None:
                batch_size_test = batch_size
            if is_test:
                batch_size_ = batch_size_test
            else:
                batch_size_ = batch_size
            if batch_size_ > len(sub_dataset):
                batch_size_ = len(sub_dataset)
            data_loader = torch.utils.data.DataLoader(
                sub_dataset,
                batch_size=batch_size_,
                shuffle= ~is_test,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(custom_collate_fn, x_collector=padding_collector),
            )
        else:
            sampler = get_dataset_sampler(dataset, split, batch_size=batch_size,
                                          num_batches_per_epoch=num_batches_per_epoch)
            data_loader = torch.utils.data.DataLoader(
                sub_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(custom_collate_fn, x_collector=padding_collector),
                sampler=sampler,
            )
        all_loader.append(data_loader)
    return tuple(all_loader)
