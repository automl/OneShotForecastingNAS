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
from tsf_oneshot.training.utils import get_optimizer, get_lr_scheduler

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
    elif dataset_type == 'PEMS':
        data_info, split_begin, split_end, (border1s, border2s) = get_PEMS_dataset.get_test_dataset(dataset_root_path,
                                                                                                    dataset_name=dataset_name,
                                                                                                    file_name=cfg.benchmark.file_name,
                                                                                                    series_type=cfg.benchmark.series_type,
                                                                                                    window_size=int(cfg.benchmark.dataloader.window_size),
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

    num_targets = dataset.num_targets

    n_time_features = len(dataset.time_feature_transform)
    d_input_past = len(dataset.lagged_value) * num_targets + n_time_features
    # d_input_future = len(dataset.lagged_value) * num_targets + n_time_features
    d_input_future = n_time_features
    d_output = dataset.num_targets

    # This value is used to initialize the networks
    search_sample_interval = cfg.benchmark.dataloader.get('search_sample_interval', 1)
    #search_sample_interval = 1
    window_size_raw = window_size
    window_size = (window_size - 1) // search_sample_interval + 1
    n_prediction_steps = (dataset.n_prediction_steps - 1) // search_sample_interval + 1

    model_path = Path(cfg.model_dir) / device / f'{dataset_type}' / cfg.benchmark.search_dataset_name / model_name / str(seed)
    if (model_path / 'OptModel' / f'opt_arch_weights.pth').exists():
        saved_data_info = torch.load(model_path / 'OptModel' / f'opt_arch_weights.pth')
    elif (model_path / 'Model' / 'model_weights.pth').exists():
        saved_data_info = torch.load(model_path / 'Model' / 'model_weights.pth')
    else:
        raise ValueError('No optimal model is found. Please ensure that you have call train_and_eval to search for the '
                         'optimal architectures!')

    if model_type == 'seq':
        operations_encoder, has_edges_encoder = get_optimized_archs(saved_data_info, 'arch_p_encoder', 'mask_encoder')
        operations_decoder, has_edges_decoder = get_optimized_archs(saved_data_info, 'arch_p_decoder', 'mask_decoder')
        decoder_choice, _ = get_optimized_archs(saved_data_info, 'arch_p_decoder_choices', 'mask_decoder_choices')
        head_idx, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_heads')

        del saved_data_info

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs = cfg_model.get('model_kwargs', {})
        ops_kwargs['mlp_mix'] = {
                                    'forecasting_horizon': n_prediction_steps,
                                    'window_size': window_size,
                                }
        ops_kwargs['transformer'] = {
            'forecasting_horizon': n_prediction_steps,
            'window_size': window_size,
        }
        heads_kwargs = cfg_model.get('heads_kwargs', {})

        head_idx = head_idx[0]
        HEAD = list(cfg.model.HEADs)[head_idx]

        decoder_choice = decoder_choice[0]
        DECODER = list(cfg.model.DECODERS)[decoder_choice]
        net_init_kwargs = {
            'd_input_past': d_input_past,
            'window_size': window_size,
            'forecasting_horizon': n_prediction_steps,
            'OPS_kwargs': ops_kwargs,
            'd_input_future': d_input_future,
            'd_model': int(cfg.model.d_model),
            'd_output': d_output,
            'n_cells': int(cfg.model.n_cells),
            'n_nodes': int(cfg.model.n_nodes),
            'operations_encoder': operations_encoder.tolist(),
            'has_edges_encoder': has_edges_encoder,
            'operations_decoder': operations_decoder.tolist(),
            'has_edges_decoder': has_edges_decoder,
            'n_cell_input_nodes': int(cfg.model.n_cell_input_nodes),
            'PRIMITIVES_encoder': list(cfg.model.PRIMITIVES_encoder),
            'PRIMITIVES_decoder': list(cfg.model.PRIMITIVES_decoder),
            'DECODER': DECODER,
            'HEAD': HEAD,
            'HEADs_kwargs': heads_kwargs,
        }
        model = SampledNet(**net_init_kwargs)

    elif model_type == 'flat':
        operations_encoder, has_edges_encoder = get_optimized_archs(saved_data_info, 'arch_p_encoder', 'mask_encoder')
        head_idx, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_heads')
        del saved_data_info
        torch.cuda.empty_cache()

        head_idx = head_idx[0]
        HEAD = list(cfg.model.HEADs)[head_idx]

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs = cfg_model.get('model_kwargs', {})
        heads_kwargs = cfg_model.get('heads_kwargs', {})

        net_init_kwargs = dict(
            window_size=window_size,
            forecasting_horizon=n_prediction_steps,
            d_output=d_output,
            OPS_kwargs=ops_kwargs,
            n_cells=int(cfg.model.n_cells),
            n_nodes=int(cfg.model.n_nodes),
            operations_encoder=operations_encoder.tolist(),
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
        decoder_choice_seq, _ = get_optimized_archs(saved_data_info, 'arch_p_decoder_choices_seq',
                                                    'mask_decoder_choices_seq')
        head, _ = get_optimized_archs(saved_data_info, 'arch_p_heads', 'mask_heads')

        nets_weights = saved_data_info['arch_p_nets'].tolist()[0]

        del saved_data_info

        head_idx = head[0]
        HEAD = list(cfg.model.HEADs)[head_idx]

        decoder_choice_seq = decoder_choice_seq[0]
        DECODER_seq = list(cfg.model.seq_model.DECODERS)[decoder_choice_seq]
       
        if DECODER_seq == 'linear':
            search_sample_interval = 1
            window_size = window_size_raw
            n_prediction_steps =int(dataset.n_prediction_steps)

        cfg_model = omegaconf.OmegaConf.to_container(cfg.model, resolve=True)

        ops_kwargs_seq = cfg_model['seq_model'].get('ops_kwargs', {})
        ops_kwargs_seq.update(cfg_model['seq_model'].get('model_kwargs', {}))

        ops_kwargs_flat = cfg_model['flat_model'].get('ops_kwargs', {})
        heads_kwargs_seq = cfg_model['seq_model'].get('head_kwargs', {})

        ops_kwargs_flat.update(cfg_model['flat_model'].get('model_kwargs', {}))
        heads_kwargs_flat = cfg_model['flat_model'].get('head_kwargs', {})

        if model_type == 'mixed_concat':
            d_input_future = num_targets + n_time_features
        net_init_kwargs = dict(d_input_past=d_input_past,
                               d_input_future=d_input_future,
                               d_output=d_output,
                               d_model=int(cfg.model.seq_model.d_model),
                               n_cells_seq=int(cfg.model.seq_model.n_cells),
                               n_nodes_seq=int(cfg.model.seq_model.n_nodes),
                               n_cell_input_nodes_seq=int(cfg.model.seq_model.n_cell_input_nodes),
                               operations_encoder_seq=operations_encoder_seq.tolist(),
                               has_edges_encoder_seq=has_edges_encoder_seq,
                               operations_decoder_seq=operations_decoder_seq.tolist(),
                               has_edges_decoder_seq=has_edges_decoder_seq,
                               PRIMITIVES_encoder_seq=list(cfg.model.seq_model.PRIMITIVES_encoder),
                               PRIMITIVES_decoder_seq=list(cfg.model.seq_model.PRIMITIVES_decoder),
                               DECODER_seq=DECODER_seq,

                               OPS_kwargs_seq=ops_kwargs_seq,
                               backcast_loss_ration_seq=float(cfg.model.seq_model.get('backcast_loss_ration', 0.0)),

                               window_size=window_size,
                               forecasting_horizon=n_prediction_steps,
                               n_cells_flat=int(cfg.model.flat_model.n_cells),
                               n_nodes_flat=int(cfg.model.flat_model.n_nodes),
                               n_cell_input_nodes_flat=int(cfg.model.flat_model.n_cell_input_nodes),
                               operations_encoder_flat=operations_encoder_flat.tolist(),
                               has_edges_encoder_flat=has_edges_encoder_flat,
                               PRIMITIVES_encoder_flat=list(cfg.model.flat_model.PRIMITIVES_encoder),
                               OPS_kwargs_flat=ops_kwargs_flat,
                               nets_weights=nets_weights,
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
    n_pars = 0

    w_optim_groups = model.get_weight_optimizer_parameters(cfg.w_optimizer_eval.weight_decay)
    w_optimizer = get_optimizer(cfg_optimizer=cfg.w_optimizer_eval, optim_groups=w_optim_groups, wd_in_p_groups=True)

    lr_scheduler = get_lr_scheduler(optimizer=w_optimizer, cfg_lr_scheduler=cfg.lr_scheduler_eval,
                                    steps_per_epoch=len(train_data_loader),
                                    )

    target_scaler = TargetScaler(cfg.train.targe_scaler)

    trainer = SampledForecastingNetTrainer(
        model=model,
        w_optimizer=w_optimizer,
        lr_scheduler_w=lr_scheduler,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        test_loader=test_data_loader,
        window_size=window_size_raw,
        n_prediction_steps=dataset.n_prediction_steps,
        lagged_values=dataset.lagged_value,
        sample_interval=search_sample_interval,
        target_scaler=target_scaler,
        grad_clip=cfg.train.grad_clip,
        device=torch.device('cuda'),
        amp_enable=cfg.train.amp_enable
    )

    early_stopping = EarlyStopping(25)

    epoch_start = 0
    #if (out_path / 'SampledNet' / 'Model').exists():
    #    epoch_start = trainer.load(out_path, model=model, w_optimizer=w_optimizer, lr_scheduler_w=lr_scheduler)
    for epoch in range(epoch_start, cfg.train.n_epochs_eval):
        val_res, test_res = trainer.train_epoch(epoch)

        do_early_stopping = early_stopping(val_res, test_res, epoch)
        if do_early_stopping:
            break
        trainer.save(out_path, epoch=epoch)

        with open(out_path / 'eval_res.json', 'w') as f:
            json.dump(test_res, f)

    print(f"best val loss: {early_stopping.best_val_loss},"
          f"best test loss: {early_stopping.best_test_loss}")
    res_all = {'val': early_stopping.val_ht,
               'test': early_stopping.test_ht,
               'best_val': early_stopping.best_val_loss,
               'best_test': early_stopping.best_test_loss}
    with open(out_path / 'eval_res.json', 'w') as f:
        json.dump(res_all, f)

if __name__ == '__main__':
    main()
