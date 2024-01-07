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
from tsf_oneshot.networks.sampled_net import SampledNet, SampledFlatNet, MixedParallelSampledNet, MixedConcatSampledNet

import torch.multiprocessing

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
        archp = saved_data_info[key].cpu().argmax(-1).tolist()
        has_edges = [True] * len(archp)
    else:
        mask = mask.cpu()
        archp = (saved_data_info[key].cpu() + mask).argmax(-1).tolist()
        has_edges = torch.isinf(mask).all(1).tolist()
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

    window_size = int(cfg.benchmark.dataloader.window_size)
    split = [np.arange(b1, b2 - dataset.n_prediction_steps) for b1, b2 in zip(border1s, border2s)]

    split = [
        np.arange(window_size - 1, border2s[0] - dataset.n_prediction_steps),
        np.arange(border1s[1] - 1, border2s[1] - dataset.n_prediction_steps),
        np.arange(border1s[2] - 1, border2s[2] - dataset.n_prediction_steps),

    ]

    train_data_loader, val_data_loader, test_data_loader = get_dataloader(
        dataset=dataset, splits=split, batch_size=32,
        #num_batches_per_epoch=cfg.benchmark.data_loader.num_batches_per_epoch,
        num_batches_per_epoch=500,
        #num_batches_per_epoch=None,
        window_size=window_size,
        is_test_sets=[False, True, True],
        batch_size_test=32,
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
        operations_encoder, has_edges_encoder = get_optimized_archs(saved_data_info, 'arch_p_encoder', 'mask_encoder')
        operations_decoder, has_edges_decoder = get_optimized_archs(saved_data_info, 'arch_p_decoder', 'mask_decoder')
        head_idx, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_head')
        del saved_data_info

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs = cfg_model.get('model_kwargs', {})
        ops_kwargs['mlp_mix'] = {
                                    'forecasting_horizon': dataset.n_prediction_steps,
                                    'window_size': window_size,
                                },
        heads_kwargs = cfg_model.get('heads_kwargs', {})

        head_idx = head_idx[0].item()
        HEAD = list(cfg.model.HEADs)[head_idx]
        net_init_kwargs = {
            'd_input_past': d_input_past,
            'OPS_kwargs': ops_kwargs,
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
            'HEADs_kwargs': heads_kwargs,
        }
        model = SampledNet(**net_init_kwargs)

    elif model_type == 'flat':
        operations_encoder, has_edges_encoder = get_optimized_archs(saved_data_info, 'arch_p_encoder', 'mask_encoder')
        head_idx, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_head')
        del saved_data_info
        torch.cuda.empty_cache()

        head_idx = head_idx[0].item()
        HEAD = list(cfg.model.HEADs)[head_idx]

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs = cfg_model.get('model_kwargs', {})
        heads_kwargs = cfg_model.get('heads_kwargs', {})

        net_init_kwargs = dict(
            window_size=window_size,
            forecasting_horizon=dataset.n_prediction_steps,
            d_output=d_output,
            OPS_kwargs=ops_kwargs,
            n_cells=int(cfg.model.n_cells) * 2,
            n_nodes=int(cfg.model.n_nodes),
            operations_encoder=operations_encoder,
            has_edges_encoder=has_edges_encoder,
            n_cell_input_nodes=int(cfg.model.n_cell_input_nodes),
            PRIMITIVES_encoder=list(cfg.model.PRIMITIVES_encoder),
            HEAD=HEAD,
            HEADs_kwargs=heads_kwargs,
            backcast_loss_ration=float(cfg.model.get('backcast_loss_ration', 0.0))
        )
        model = SampledFlatNet(**net_init_kwargs)
    elif model_type.startswith('mixed'):
        operations_encoder_seq, has_edges_encoder_seq = get_optimized_archs(
            saved_data_info, 'arch_p_encoder_seq', 'mask_encoder_seq'
        )
        operations_decoder_seq, has_edges_decoder_seq = get_optimized_archs(
            saved_data_info, 'arch_p_decoder_seq', 'mask_decoder_seq'
        )
        operations_encoder_flat, has_edges_encoder_flat = get_optimized_archs(
            saved_data_info, 'arch_p_encoder_flat', 'mask_encoder_flat'
        )
        head, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_head')

        del saved_data_info

        head_idx = head[0]

        HEAD = list(cfg.model.HEADs)[head_idx]

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs_seq = cfg_model['seq_model'].get('model_kwargs', {})


        ops_kwargs_seq['mlp_mix'] = {
                    'forecasting_horizon': dataset.n_prediction_steps,
                    'window_size': window_size,
                }
        heads_kwargs_seq = cfg_model['flat_model'].get('head_kwargs', {})

        ops_kwargs_flat = cfg_model['flat_model'].get('model_kwargs', {})
        heads_kwargs_flat = cfg_model['flat_model'].get('head_kwargs', {})

        if model_type == 'mixed_concat':
            d_input_future = d_input_past
        net_init_kwargs = dict(d_input_past=d_input_past,
                               d_input_future=d_input_future,
                               d_output=d_output,
                               d_model=int(cfg.model.seq_model.d_model),
                               n_cells_seq=int(cfg.model.seq_model.n_cells),
                               n_nodes_seq=int(cfg.model.seq_model.n_nodes),
                               n_cell_input_nodes_seq=int(cfg.model.seq_model.n_cell_input_nodes),
                               operations_encoder_seq=operations_encoder_seq,
                               has_edges_encoder_seq=has_edges_encoder_seq,
                               operations_decoder_seq=operations_decoder_seq,
                               has_edges_decoder_seq=has_edges_decoder_seq,
                               PRIMITIVES_encoder_seq=list(cfg.model.seq_model.PRIMITIVES_encoder),
                               PRIMITIVES_decoder_seq=list(cfg.model.seq_model.PRIMITIVES_decoder),

                               OPS_kwargs_seq=ops_kwargs_seq,
                               backcast_loss_ration_seq=float(cfg.model.seq_model.get('backcast_loss_ration', 0.0)),

                               window_size=window_size,
                               forecasting_horizon=dataset.n_prediction_steps,
                               n_cells_flat=int(cfg.model.flat_model.n_cells),
                               n_nodes_flat=int(cfg.model.flat_model.n_nodes),
                               n_cell_input_nodes_flat=int(cfg.model.flat_model.n_cell_input_nodes),
                               operations_encoder_flat=operations_encoder_flat,
                               has_edges_encoder_flat=has_edges_encoder_flat,
                               PRIMITIVES_encoder_flat=list(cfg.model.flat_model.PRIMITIVES_encoder),
                               OPS_kwargs_flat=ops_kwargs_flat,
                               HEAD=HEAD,
                               HEADs_kwargs_seq=heads_kwargs_seq,
                               HEADs_kwargs_flat=heads_kwargs_flat,
                               backcast_loss_ration_flat=float(cfg.model.flat_model.get('backcast_loss_ration', 0.0)),
                               )
        if model_type == 'mixed_concat':
            model = MixedConcatSampledNet(**net_init_kwargs)
        elif model_type == 'mixed_parallel':
            model = MixedParallelSampledNet(**net_init_kwargs)
    else:
        raise NotImplementedError

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
    elif cfg.w_optimizer.type == 'adam':
        w_optimizer = torch.optim.Adam(
            params=w_optim_groups,
            lr=cfg.w_optimizer.lr,
            betas=(cfg.w_optimizer.beta1, cfg.w_optimizer.beta2),
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
        val_loader=val_data_loader,
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
            json.dump(eval_res, f)


if __name__ == '__main__':
    main()
