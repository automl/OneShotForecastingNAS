import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import shutil

from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from datasets import get_LTSF_dataset, get_monash_dataset
from autoPyTorch.datasets.resampling_strategy import (
    HoldoutValTypes,
    ResamplingStrategies
)
from autoPyTorch.pipeline.components.training.data_loader.time_series_forecasting_data_loader import TimeSeriesForecastingDataLoader


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: omegaconf.DictConfig):
    dataset_type = cfg.benchmark.type
    dataset_name = cfg.benchmark.name
    dataset_root_path = Path(cfg.benchmark.dataset_root) / dataset_type

    resampling_strategy: ResamplingStrategies = HoldoutValTypes.time_series_hold_out_validation,
    resampling_strategy_args = None,


    # Create a validator object to make sure that the data provided by
    # the user matches the autopytorch requirements
    input_validator = TimeSeriesForecastingInputValidator(
        is_classification=False,
    )

    if dataset_type == 'monash':
        data_info, y_test = get_monash_dataset.get_train_dataset(dataset_root_path,
                                                                 dataset_name=dataset_name,
                                                                 external_forecast_horizon=cfg.benchmark.external_forecast_horizon,
                                                                 file_name=cfg.benchmark.file_name,
                                                                 )
    elif dataset_type == 'LTSF':
        data_info = get_LTSF_dataset.get_train_dataset(dataset_root_path, dataset_name=dataset_name,
                                                       file_name=cfg.benchmark.file_name,
                                                       series_type=cfg.benchmark.series_type,
                                                       do_normalization=cfg.benchmark.do_normalization,
                                                       forecasting_horizon=cfg.benchmark.external_forecast_horizon,
                                                       make_dataset_uni_variant=cfg.benchmark.get("make_dataset_uni_variant", False),
                                                       flag='train')
    else:
        raise NotImplementedError


    data_y = data_info['y_train']
    start_time = data_info['start_times']
    n_prediction_steps = data_info['n_prediction_steps']
    X_train = None

    # Fit an input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    input_validator.fit(
        X_train=X_train,
        y_train=data_y,
        start_times=start_time,
    )

    dataset = TimeSeriesForecastingDataset(
        X=X_train,
        Y=data_y,
        dataset_name=dataset_name,
        freq=data_info['freq'],
        start_times=start_time,
        validator=input_validator,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        n_prediction_steps=n_prediction_steps,
    )

    train_split, test_split = dataset.splits[0]
    mase_coefficient = np.ones([len(test_split), dataset.num_targets])



if __name__ == '__main__':
    main()
