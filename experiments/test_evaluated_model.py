from tsf_oneshot.networks.sampled_net import SampledNet, SampledFlatNet
import numpy as np
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
from datasets import get_LTSF_dataset, get_monash_dataset
from datasets.get_data_loader import get_forecasting_dataset, get_dataloader, regenerate_splits

from tsf_oneshot.training.samplednet_trainer import SampledForecastingNetTrainer
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # OTHERWISE Conv1D with dilation will be too slow?
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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

    out_path = Path(cfg.model_dir) / device / dataset_type / dataset_name / model_name / str(seed)
    if not out_path.exists():
        os.makedirs(out_path, exist_ok=True)

    if dataset_type == 'monash':
        data_info, y_test = get_monash_dataset.get_train_dataset(dataset_root_path,
                                                                 dataset_name=dataset_name,
                                                                 file_name=cfg.benchmark.file_name,
                                                                 external_forecast_horizon=cfg.benchmark.get(
                                                                     'external_forecast_horizon', None)
                                                                 )
    elif dataset_type == 'LTSF':
        data_info, split_begin, split_end = get_LTSF_dataset.get_test_dataset(dataset_root_path,
                                                                              dataset_name=dataset_name,
                                                                              file_name=cfg.benchmark.file_name,
                                                                              series_type=cfg.benchmark.series_type,
                                                                              do_normalization=cfg.benchmark.do_normalization,
                                                                              forecasting_horizon=cfg.benchmark.external_forecast_horizon,
                                                                              make_dataset_uni_variant=cfg.benchmark.get(
                                                                                  "make_dataset_uni_variant", False),
                                                                              flag='test')
    else:
        raise NotImplementedError

    dataset = get_forecasting_dataset(dataset_name=dataset_name, **data_info)
    dataset.lagged_value = [0]  # + get_lags_for_frequency(dataset.freq, num_default_lags=1)
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

    window_size = int(cfg.dataloader.window_size)

    split = [np.arange(split_begin - dataset.n_prediction_steps), np.arange(split_begin, split_end)]

    train_data_loader, test_data_loader = get_dataloader(
        dataset=dataset, splits=split, batch_size=32,
        num_batches_per_epoch=cfg.dataloader.num_batches_per_epoch,
        window_size=window_size,
        is_test_sets=[False, True],
        batch_size_test=8,
    )
    num_targets = dataset.num_targets

    n_time_features = len(dataset.time_feature_transform)
    d_input_past = len(dataset.lagged_value) * num_targets + n_time_features
    # d_input_future = len(dataset.lagged_value) * num_targets + n_time_features
    d_input_future = n_time_features
    d_output = dataset.num_targets

    # TODO check what data to pass
    saved_data_info = torch.load(out_path / 'Model' / 'model_weights.pth')
    if model_type == 'seq':
        archp_encoder = saved_data_info['arch_p_encoder'].cpu()
        archp_decoder = saved_data_info['arch_p_decoder'].cpu()
        archp_head = saved_data_info['arch_p_heads'].cpu()
        del saved_data_info
        torch.cuda.empty_cache()
        operations_encoder = archp_encoder.argmax(-1).tolist()
        has_edges_encoder = [True] * len(operations_encoder)

        operations_decoder = archp_decoder.argmax(-1).tolist()
        has_edges_decoder = [True] * len(archp_decoder)

        head_idx = archp_head.argmax(-1)[0].item()
        HEAD = list(cfg.model.HEADS)[head_idx]
        net_init_kwargs = {
            'd_input_past': d_input_past,
            'OPS_kwargs': {
                'mlp_mix': {
                    'forecasting_horizon': dataset.n_prediction_steps,
                    'window_size': window_size,
                }
            },
            'd_input_future': d_input_future,
            'd_model': int(cfg.model.d_model),
            'd_output': d_output,
            'n_cells': int(cfg.model.n_cells) * 2,
            'n_nodes': int(cfg.model.n_nodes),
            'operations_encoder': operations_encoder,
            'has_edges_encoder': has_edges_encoder,
            'operations_decoder': operations_decoder,
            'has_edges_decoder': has_edges_decoder,
            'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
            'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
            'PRIMITIVES_decoder': list(cfg.model.PRIMITIVES_decoder),
            'HEAD': HEAD,
            'HEADS_kwargs': {},
        }
        model = SampledNet(**net_init_kwargs)

    elif model_type == 'flat':
        archp_encoder = saved_data_info['arch_p_encoder'].cpu()
        archp_head = saved_data_info['arch_p_heads'].cpu()
        del saved_data_info
        torch.cuda.empty_cache()
        operations_encoder = archp_encoder.argmax(-1).tolist()
        has_edges_encoder = [True] * len(operations_encoder)
        head_idx = archp_head.argmax(-1)[0].item()
        HEAD = list(cfg.model.HEADS)[head_idx]

        net_init_kwargs = {
            'window_size': window_size,
            'forecasting_horizon': dataset.n_prediction_steps,
            'OPS_kwargs': {},
            'n_cells': int(cfg.model.n_cells) * 2,
            'n_nodes': int(cfg.model.n_nodes),
            'operations_encoder': operations_encoder,
            'has_edges_encoder': has_edges_encoder,
            'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
            'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
            'HEAD': HEAD,
            'HEADS_kwargs': {},
        }
        model = SampledFlatNet(**net_init_kwargs)
    else:
        raise NotImplementedError

    """
    net_init_kwargs = {
        'd_input_past': d_input_past,
        'OPS_kwargs': {
            'mlp_mix': {
                'forecasting_horizon': dataset.n_prediction_steps,
                'window_size': window_size,
            }
        },
        'd_input_future': d_input_future,
        'd_model': int(cfg.model.d_model),
        'd_output': d_output,
        'n_cells': int(cfg.model.n_cells),
        'n_nodes': int(cfg.model.n_nodes),
        'operations_encoder': operations_encoder,
        'has_edges_encoder': has_edges_encoder,
        'operations_decoder': operations_decoder,
        'has_edges_decoder': has_edges_decoder,
        'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
        'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
        'PRIMITIVES_decoder': list(cfg.model.PRIMITIVES_decoder),
        'HEAD': HEAD,
        'HEADS_kwargs': {},
    }
    """

    # model = SampledNet(**net_init_kwargs)

    w_optim_groups = model.get_weight_optimizer_parameters(cfg.w_optimizer.weight_decay)

    if cfg.w_optimizer.type == 'adamw':
        w_optimizer = torch.optim.AdamW(
            params=w_optim_groups,
            lr=cfg.w_optimizer.lr,
            betas=(cfg.w_optimizer.beta1, cfg.w_optimizer.beta2),
        )
    elif cfg.w_optimizer.type == 'sgd':
        w_optimizer = torch.optim.SGD(
            params=w_optim_groups,
            lr=cfg.w_optimizer.lr,
            momentum=cfg.w_optimizer.momentum,
        )
    else:
        raise NotImplementedError

    if cfg.lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=w_optimizer,
            **cfg.lr_scheduler
        )
    else:
        raise NotImplementedError

    target_scaler = TargetScaler(cfg.train.targe_scaler)

    trainer = SampledForecastingNetTrainer(
        model=model,
        w_optimizer=w_optimizer,
        lr_scheduler_w=lr_scheduler,
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        window_size=window_size,
        n_prediction_steps=dataset.n_prediction_steps,
        lagged_values=dataset.lagged_value,
        target_scaler=target_scaler,
        grad_clip=cfg.train.grad_clip,
        device=torch.device('cuda'),
        amp_enable=cfg.train.amp_enable
    )

    epoch_start = 0
    if (out_path / 'SampledNet' / 'Model').exists():
        epoch_start = trainer.load(out_path, model=model, w_optimizer=w_optimizer, lr_scheduler_w=lr_scheduler)
    for epoch in range(epoch_start, cfg.train.n_epochs):
        eval_res = trainer.train_epoch(epoch)
        trainer.save(out_path, epoch=epoch)

        with open(out_path / 'eval_res.json', 'w') as f:
            json.dump(eval_res)


if __name__ == '__main__':
    main()
