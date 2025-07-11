import copy
import os
from pathlib import Path
import random
import json

import hydra
import numpy as np
import omegaconf
import torch
import wandb

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from datasets import get_LTSF_dataset, get_monash_dataset, get_PEMS_dataset
from datasets.get_data_loader import get_forecasting_dataset, get_dataloader, regenerate_splits

from tsf_oneshot.training.samplednet_trainer import SampledForecastingNetTrainer, EarlyStopping
from tsf_oneshot.networks.sampled_net import SampledNet, SampledFlatNet, MixedParallelSampledNet, MixedConcatSampledNet
from tsf_oneshot.training.utils import get_optimizer, get_lr_scheduler, ForecastingPreprocessor

import torch.multiprocessing

from tsf_oneshot.naslib_zcevaluator.zc_predcitor import ZeroCost

torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Otherwise, Conv1D with dilation will be too slow?
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_optimized_archs(saved_data_info: dict[str, torch.Tensor], key: str, mask_key: str):
    mask = saved_data_info.get(mask_key, None)
    if mask is None:
        archp = saved_data_info[key].cpu().argmax(-1)
        has_edges = [True] * len(archp)
    else:
        mask = mask.cpu()
        archp = (saved_data_info[key].cpu() + mask).argmax(-1)
        has_edges = (~torch.isinf(mask).all(1)).tolist()
    return archp, has_edges


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: omegaconf.DictConfig):
    dataset_type = cfg.benchmark.type
    dataset_name = cfg.benchmark.name
    dataset_root_path = Path(cfg.benchmark.dataset_root) / dataset_type
    seed = cfg.seed

    seed_everything(seed)

    cfg.wandb.project = f'{cfg.wandb.project}_{seed}_eval'
    cfg.wandb.group = f'{cfg.wandb.group}_eval'

    device = 'cuda'

    search_type = cfg.model.type
    model_type = cfg.model.model_type
    model_name = f'{model_type}_{search_type}'

    wandb.init(**cfg.wandb,
               tags=[f'seed_{seed}', model_name],
               )

    out_path = Path(cfg.model_dir) / device / f'{dataset_type}' / dataset_name / model_name / str(seed)

    model_meta_info_path = out_path / 'SampledNet' / 'Model' / 'meta_info.json'
    with open(model_meta_info_path, 'r') as f:
        model_meta_info = json.load(f)

    model = MixedConcatSampledNet(**model_meta_info)

    # if not out_path.exists():
    #    os.makedirs(out_path, exist_ok=True)

    if dataset_type == 'monash':
        data_info, y_test = get_monash_dataset.get_train_dataset(dataset_root_path,
                                                                 dataset_name=dataset_name,
                                                                 file_name=cfg.benchmark.file_name,
                                                                 external_forecast_horizon=cfg.benchmark.get(
                                                                     'external_forecast_horizon', None)
                                                                 )
    elif dataset_type == 'LTSF':
        data_info, split_begin, split_end, (border1s, border2s) = get_LTSF_dataset.get_test_dataset(dataset_root_path,
                                                                                                    dataset_name=dataset_name,
                                                                                                    file_name=cfg.benchmark.file_name,
                                                                                                    series_type=cfg.benchmark.series_type,
                                                                                                    do_normalization=cfg.benchmark.do_normalization,
                                                                                                    forecasting_horizon=cfg.benchmark.external_forecast_horizon,
                                                                                                    make_dataset_uni_variant=cfg.benchmark.get(
                                                                                                        "make_dataset_uni_variant",
                                                                                                        False),
                                                                                                    flag='test')
    elif dataset_type == 'PEMS':
        data_info, split_begin, split_end, (border1s, border2s) = get_PEMS_dataset.get_test_dataset(dataset_root_path,
                                                                                                    dataset_name=dataset_name,
                                                                                                    file_name=cfg.benchmark.file_name,
                                                                                                    series_type=cfg.benchmark.series_type,
                                                                                                    window_size=int(
                                                                                                        cfg.benchmark.dataloader.window_size),
                                                                                                    do_normalization=cfg.benchmark.do_normalization,
                                                                                                    forecasting_horizon=cfg.benchmark.external_forecast_horizon,
                                                                                                    make_dataset_uni_variant=cfg.benchmark.get(
                                                                                                        "make_dataset_uni_variant",
                                                                                                        False),
                                                                                                    flag='test')
    else:
        raise NotImplementedError

    dataset = get_forecasting_dataset(dataset_name=dataset_name, **data_info)
    dataset.lagged_value = [0]
    """
    if dataset.freq == '1H' and dataset.n_prediction_steps > 168:
        base_window_size = int(168 * cfg.dataloader.window_size_coefficient)
    else:
        if cfg.benchmark.get('base_window_size', None) is None:
            base_window_size = int(np.ceil(dataset.base_window_size))
        else:
            base_window_size = cfg.benchmark.base_window_size
    window_size = int(base_window_size * cfg.dataloader.window_size_coefficient)
    """

    window_size = int(cfg.benchmark.dataloader.window_size)

    split_ms = [
        np.arange(window_size - 1, border2s[0] - dataset.n_prediction_steps),
        np.arange(border1s[1] - 1, border2s[1] - dataset.n_prediction_steps),
        np.arange(border1s[2] - 1, border2s[2] - dataset.n_prediction_steps),
    ]
    split = regenerate_splits(dataset, val_share=None, splits_ms=split_ms)
    train_data_loader, val_data_loader, test_data_loader = get_dataloader(
        dataset=dataset, splits=split, batch_size=cfg.benchmark.dataloader.batch_size,
        window_size=window_size,
        is_test_sets=[False, True, True],
        batch_size_test=128,
    )

    # This value is used to initialize the networks
    search_sample_interval = cfg.benchmark.dataloader.get('search_sample_interval', 1)
    # search_sample_interval = 1
    window_size_raw = window_size
    window_size = (window_size - 1) // search_sample_interval + 1
    n_prediction_steps = (dataset.n_prediction_steps - 1) // search_sample_interval + 1

    target_scaler = TargetScaler(cfg.train.targe_scaler)

    preprocessor = ForecastingPreprocessor(window_size=window_size,
                                           n_prediction_steps=n_prediction_steps,
                                           lagged_values=dataset.lagged_value,
                                           sample_interval=search_sample_interval,
                                           target_scaler=target_scaler,
                                           device=device
                                           )

    all_zc_scores = {}
    for zc_evaluator_type in [
        'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
        'flops', 'params', 'nwot', 'plain', 'snip', 'synflow', 'zen'
        ]:
        evaluator = ZeroCost(method_type=zc_evaluator_type)
        score = evaluator.query(model, train_data_loader, preprocessor,)
        print(zc_evaluator_type)
        print(score)
        all_zc_scores[zc_evaluator_type] = score

    with open(out_path / 'zc_score.json', 'w') as f:
        json.dump(all_zc_scores, f)

if __name__ == '__main__':
    main()
