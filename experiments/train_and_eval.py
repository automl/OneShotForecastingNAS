import os
from pathlib import Path
import random

import hydra
import numpy as np
import omegaconf
import torch
import wandb

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from datasets import get_LTSF_dataset, get_monash_dataset
from datasets.get_data_loader import get_forecasting_dataset, get_dataloader, regenerate_splits

from tsf_oneshot.networks.architect import Architect
from tsf_oneshot.networks.network_controller import (
    ForecastingGDASNetworkController,
    ForecastingDARTSNetworkController,
    ForecastingDARTSFlatNetworkController,
    ForecastingGDASFlatNetworkController,
    ForecastingDARTSMixedParallelNetController,
    ForecastingGDASMixedParallelNetController,
    ForecastingDARTSMixedConcatNetController,
    ForecastingGDASMixedConcatNetController,
)
from tsf_oneshot.cells.encoders.components import TCN_DEFAULT_KERNEL_SIZE

from tsf_oneshot.training.trainer import ForecastingTrainer, ForecastingDARTSSecondOrderTrainer
from tsf_oneshot.training.utils import get_optimizer


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Otherwise, Conv1D with dilation will be too slow
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: omegaconf.DictConfig):
    dataset_type = cfg.benchmark.type
    dataset_name = cfg.benchmark.name
    dataset_root_path = Path(cfg.benchmark.dataset_root) / dataset_type
    seed = cfg.seed

    seed_everything(seed)

    cfg.wandb.project = f'{cfg.wandb.project}_{seed}'

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
        data_info = get_LTSF_dataset.get_train_dataset(dataset_root_path, dataset_name=dataset_name,
                                                       file_name=cfg.benchmark.file_name,
                                                       series_type=cfg.benchmark.series_type,
                                                       do_normalization=cfg.benchmark.do_normalization,
                                                       forecasting_horizon=cfg.benchmark.external_forecast_horizon,
                                                       make_dataset_uni_variant=cfg.benchmark.get(
                                                           "make_dataset_uni_variant", False),
                                                       flag='train')
    else:
        raise NotImplementedError

    dataset = get_forecasting_dataset(dataset_name=dataset_name, **data_info)
    dataset.lagged_value = [0]  # + get_lags_for_frequency(dataset.freq, num_default_lags=1)

    val_share: float = cfg.val_share
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
    start_idx = window_size + max(dataset.lagged_value)
    splits_new = regenerate_splits(dataset, val_share=val_share, start_idx=start_idx, strategy='cv')

    # indices = np.arange(start_idx, len(dataset))

    # splits_new = [indices, indices]

    if search_type != 'darts':
        # for gdas, we need more iterations
        num_batches_per_epoch = int(cfg.benchmark.dataloader.num_batches_per_epoch) * 2
        n_epochs = int(cfg.train.n_epochs) * 2
        search_sample_interval = 1
    else:
        num_batches_per_epoch = int(cfg.benchmark.dataloader.num_batches_per_epoch)
        n_epochs = int(cfg.train.n_epochs)
        search_sample_interval = cfg.benchmark.dataloader.get('search_sample_interval', 1)
    batch_size = cfg.benchmark.dataloader.batch_size

    train_data_loader, val_data_loader = get_dataloader(
        dataset=dataset, splits=splits_new, batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        window_size=window_size,
        sample_interval=search_sample_interval
    )

    if cfg.model.get('select_with_pt', False):
        # if we would like to select architectures with perturbation: https://openreview.net/pdf?id=PKubaeJkw3
        # we need to have another validation set to determine which edge and architectures performances best
        split_val_pt_size = regenerate_splits(dataset, val_share=0.1, start_idx=start_idx, strategy='holdout')[-1]
        val_eval_loader = get_dataloader(
            dataset=dataset, splits=[split_val_pt_size], batch_size=batch_size,
            num_batches_per_epoch=None,
            is_test_sets=[True],
            window_size=window_size,
            sample_interval=search_sample_interval,
            batch_size_test=batch_size,
        )[0]
        # the number of epochs that we need to train before evaluating the actual edge
        proj_intv = cfg.model.get("proj_intv", 1)
    else:
        proj_intv = 1
        val_eval_loader = None

    # we need to adjust the values to initialize our networks
    window_size = (window_size - 1) // search_sample_interval + 1
    n_prediction_steps = (dataset.n_prediction_steps - 1) // search_sample_interval + 1

    num_targets = dataset.num_targets

    n_time_features = len(dataset.time_feature_transform)
    d_input_past = len(dataset.lagged_value) * num_targets + n_time_features
    # d_input_future = len(dataset.lagged_value) * num_targets + n_time_features
    d_input_future = n_time_features
    d_output = dataset.num_targets

    if model_type == 'seq':
        net_init_kwargs = {
            'n_cells': int(cfg.model.n_cells),
            'n_nodes': int(cfg.model.n_nodes),
            'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
            'backcast_loss_ration': float(cfg.model.get('backcast_loss_ration', 0.0))
        }
        ops_kwargs = dict(cfg.model.get('model_kwargs', {}))
        ops_kwargs['mlp_mix'] = {
                                    'forecasting_horizon': n_prediction_steps,
                                    'window_size': window_size,
                                }
        heads_kwargs = dict(cfg.model.get('heads_kwargs', {}))
        if 'tcn' in ops_kwargs:
            ops_kwargs['tcn'].update(
                {
                    'kernel_size': int(np.ceil(ops_kwargs['tcn'].get('kernel_size', TCN_DEFAULT_KERNEL_SIZE)) / search_sample_interval)
                }
            )
        else:
            ops_kwargs['tcn'] = {
                    'kernel_size': int(np.ceil(TCN_DEFAULT_KERNEL_SIZE / search_sample_interval))
            }

        net_init_kwargs.update(
            {'d_input_past': d_input_past,
             'OPS_kwargs': ops_kwargs,
             'd_input_future': d_input_future,
             'd_model': int(cfg.model.d_model),
             'd_output': d_output,
             'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
             'PRIMITIVES_decoder': list(cfg.model.PRIMITIVES_decoder),
             'HEADs': list(cfg.model.HEADs),
             'HEADs_kwargs': heads_kwargs
             }
        )
    elif model_type == 'flat':
        net_init_kwargs = {
            'n_cells': int(cfg.model.n_cells),
            'n_nodes': int(cfg.model.n_nodes),
            'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
            'd_output': d_output,
            'backcast_loss_ration': float(cfg.model.get('backcast_loss_ration', 0.0))
        }
        ops_kwargs = dict(cfg.model.get('model_kwargs', {}))
        heads_kwargs = dict(cfg.model.get('heads_kwargs', {}))
        net_init_kwargs.update(
            {'window_size': window_size,
             'forecasting_horizon': n_prediction_steps,
             'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
             'HEADs': list(cfg.model.HEADs),
             'OPS_kwargs': ops_kwargs,
             'HEADs_kwargs': heads_kwargs
             }
        )
    elif model_type.startswith('mixed'):
        if model_type == 'mixed_concat':
            d_input_future = d_input_past

        ops_kwargs_seq = dict(cfg.model.seq_model.get('model_kwargs', {}))

        if 'tcn' in ops_kwargs_seq:
            ops_kwargs_seq['tcn'].update(
                {
                    'kernel_size': int(np.ceil(ops_kwargs_seq['tcn'].get('kernel_size', TCN_DEFAULT_KERNEL_SIZE)) / search_sample_interval)
                }
            )
        else:
            ops_kwargs_seq['tcn'] = {
                    'kernel_size': int(np.ceil(TCN_DEFAULT_KERNEL_SIZE / search_sample_interval))
            }
        ops_kwargs_seq['mlp_mix'] = {
                    'forecasting_horizon': n_prediction_steps,
                    'window_size': window_size,
                }
        heads_kwargs_seq = dict(cfg.model.flat_model.get('head_kwargs', {}))

        ops_kwargs_flat = dict(cfg.model.flat_model.get('model_kwargs', {}))
        heads_kwargs_flat = dict(cfg.model.flat_model.get('head_kwargs', {}))


        net_init_kwargs = {
            'd_input_past': d_input_past,
            'd_input_future': d_input_future,
            'd_model': int(cfg.model.seq_model.d_model),
            'd_output': d_output,
            'n_cells_seq': int(cfg.model.seq_model.n_cells),
            'n_nodes_seq': int(cfg.model.seq_model.n_nodes),
            'n_cell_input_nodes_seq': int(cfg.model.seq_model.n_cell_input_nodes),
            'backcast_loss_ration_seq': float(cfg.model.seq_model.get('backcast_loss_ration', 0.0)),

            'PRIMITIVES_encoder_seq': list(cfg.model.seq_model.PRIMITIVES_encoder),
            'PRIMITIVES_decoder_seq': list(cfg.model.seq_model.PRIMITIVES_decoder),
            'OPS_kwargs_seq': ops_kwargs_seq,

            'window_size': window_size,
            'forecasting_horizon': n_prediction_steps,
            'n_cells_flat': int(cfg.model.flat_model.n_cells),
            'n_nodes_flat': int(cfg.model.flat_model.n_nodes),
            'n_cell_input_nodes_flat': int(cfg.model.flat_model.n_cell_input_nodes),
            'backcast_loss_ration_flat': float(cfg.model.flat_model.get('backcast_loss_ration', 0.0)),

            'PRIMITIVES_encoder_flat': list(cfg.model.flat_model.PRIMITIVES_encoder),

            'OPS_kwargs_flat': ops_kwargs_flat,

            'HEADs': list(cfg.model.HEADs),
            'HEADs_kwargs_seq': heads_kwargs_seq,
            'HEADs_kwargs_flat': heads_kwargs_flat,
        }
    else:
        raise NotImplementedError("Unknown model_class ")

    grad_order = cfg.model.get('grad_order', 1)

    if model_type == 'seq':
        if search_type == 'darts':
            model = ForecastingDARTSNetworkController(
                **net_init_kwargs
            )
            if grad_order == 2:
                architect = Architect(net=model, w_momentum=cfg.w_optimizer.beta1,
                                      w_weight_decay=cfg.w_optimizer.weight_decay)
        elif search_type == 'gdas':
            model = ForecastingGDASNetworkController(**net_init_kwargs)
        else:
            raise NotImplementedError
    elif model_type == 'flat':
        if search_type == 'darts':
            model = ForecastingDARTSFlatNetworkController(
                **net_init_kwargs
            )
            if grad_order == 2:
                architect = Architect(net=model, w_momentum=cfg.w_optimizer.beta1,
                                      w_weight_decay=cfg.w_optimizer.weight_decay)
        elif search_type == 'gdas':
            model = ForecastingGDASFlatNetworkController(**net_init_kwargs)
        else:
            raise NotImplementedError
    elif model_type == 'mixed_parallel':
        if search_type == 'darts':
            model = ForecastingDARTSMixedParallelNetController(
                **net_init_kwargs
            )
            if grad_order == 2:
                architect = Architect(net=model, w_momentum=cfg.w_optimizer.beta1,
                                      w_weight_decay=cfg.w_optimizer.weight_decay)
        elif search_type == 'gdas':
            model = ForecastingGDASMixedParallelNetController(**net_init_kwargs)
        else:
            raise NotImplementedError
    elif model_type == 'mixed_concat':
        if search_type == 'darts':
            model = ForecastingDARTSMixedConcatNetController(
                **net_init_kwargs
            )
            if grad_order == 2:
                architect = Architect(net=model, w_momentum=cfg.w_optimizer.beta1,
                                      w_weight_decay=cfg.w_optimizer.weight_decay)
        elif search_type == 'gdas':
            model = ForecastingGDASMixedConcatNetController(**net_init_kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown model_type {model_type}')

    optim_groups = model.get_weight_optimizer_parameters(cfg.w_optimizer.weight_decay)

    w_optimizer = get_optimizer(cfg_optimizer=cfg.w_optimizer, optim_groups=optim_groups, wd_in_p_groups=True)

    a_optimizer = get_optimizer(cfg_optimizer=cfg.a_optimizer, optim_groups=model.arch_parameters(),
                                wd_in_p_groups=False)

    if cfg.lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=w_optimizer,
            **cfg.lr_scheduler
        )
    else:
        raise NotImplementedError

    target_scaler = TargetScaler(cfg.train.targe_scaler)

    trainer_init_kwargs = dict(
        model=model,
        w_optimizer=w_optimizer,
        a_optimizer=a_optimizer,
        lr_scheduler_w=lr_scheduler,
        lr_scheduler_a=None,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        val_eval_loader=val_eval_loader,
        proj_intv=proj_intv,
        window_size=window_size,
        n_prediction_steps=n_prediction_steps,
        search_sample_interval=search_sample_interval,
        lagged_values=dataset.lagged_value,
        target_scaler=target_scaler,
        grad_clip=cfg.train.grad_clip,
        device=torch.device('cuda'),
        amp_enable=cfg.train.amp_enable
    )
    if grad_order == 2:
        trainer = ForecastingDARTSSecondOrderTrainer(
            architect=architect,
            **trainer_init_kwargs
        )
    else:
        trainer = ForecastingTrainer(
            **trainer_init_kwargs
        )
    epoch_start = 0

    if (out_path / 'Model').exists():
        epoch_start = trainer.load(out_path, model=model, w_optimizer=w_optimizer,
                                   a_optimizer=a_optimizer, lr_scheduler_w=lr_scheduler)

    # out_neg = Path(cfg.model_dir) / device / dataset_type / dataset_name / model_name / f'{seed}_w_negative_loss'
    # epoch_start = trainer.load(out_path, model=model, w_optimizer=w_optimizer,
    #                               a_optimizer=a_optimizer, lr_scheduler_w=lr_scheduler)

    for epoch in range(epoch_start, n_epochs):
        w_loss = trainer.train_epoch(epoch)
        if epoch in [29, 59, 89, 119, 149]:
            trainer.save(out_path / f'epoch_{epoch}', epoch=epoch)

        if not torch.isnan(w_loss):
            trainer.save(out_path, epoch=epoch)

            if w_loss < 0:
                out_neg = Path(
                    cfg.model_dir) / device / dataset_type / dataset_name / model_name / f'{seed}_w_negative_loss'
                if not out_neg.exists():
                    os.makedirs(out_neg, exist_ok=True)
                    trainer.save(out_neg, epoch=epoch)
        else:
            break
    if search_type == 'darts':
        trainer.pt_project(cfg)
        trainer.save(out_path, epoch=n_epochs - 1)


if __name__ == '__main__':
    main()
