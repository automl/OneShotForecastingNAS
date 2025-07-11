from enum import Enum

import omegaconf

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import get_lagged_subsequences

import torch
from tsf_oneshot.training.training_utils import scale_value


def get_optimizer(cfg_optimizer: omegaconf.DictConfig,
                  optim_groups: list[torch.nn.Parameter],
                  wd_in_p_groups: bool = False) -> torch.optim.Optimizer:
    weight_decay = 0 if wd_in_p_groups else cfg_optimizer.weight_decay
    if cfg_optimizer.type == 'sgd':
        optimizer = torch.optim.SGD(
            params=optim_groups,
            lr=cfg_optimizer.lr,
            momentum=cfg_optimizer.momentum,
            nesterov=cfg_optimizer.nesterov,
            weight_decay=weight_decay
        )
    elif cfg_optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=cfg_optimizer.lr,
            betas=(cfg_optimizer.beta1, cfg_optimizer.beta2),
            weight_decay=weight_decay
        )
    elif cfg_optimizer.type == 'adam':
        optimizer = torch.optim.Adam(
            params=optim_groups,
            lr=cfg_optimizer.lr,
            betas=(cfg_optimizer.beta1, cfg_optimizer.beta2),
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(f'Unknown Optimizer Type: {cfg_optimizer.type}')
    return optimizer


class LR_SCHEDULER_TYPE(Enum):
    epoch = 1
    batch = 2


def pad_tensor(tensor_to_be_padded: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    pad tensor to meet the required length

    Args:
         tensor_to_be_padded (torch.Tensor)
            tensor to be padded
         target_length  (int):
            target length

    Return:
        torch.Tensor:
            padded tensors
    """
    tensor_shape = tensor_to_be_padded.shape
    padding_size = [tensor_shape[0], target_length - tensor_shape[1], tensor_shape[-1]]
    tensor_to_be_padded = torch.cat([tensor_to_be_padded.new_zeros(padding_size), tensor_to_be_padded], dim=1)
    return tensor_to_be_padded



def get_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg_lr_scheduler: omegaconf.DictConfig,
        **kwargs,
):
    cfg_lr_scheduler_copy = omegaconf.OmegaConf.to_container(cfg_lr_scheduler.copy(), resolve=True)
    lr_scheduler_type = cfg_lr_scheduler_copy.pop('type')

    if lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            **cfg_lr_scheduler_copy
        )
        return lr_scheduler
    elif lr_scheduler_type == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=kwargs['steps_per_epoch'],
            **cfg_lr_scheduler_copy
        )
        return lr_scheduler
    else:
        raise NotImplementedError(f"Unknown lr scheduler: {lr_scheduler_type}")


class ForecastingPreprocessor:
    def __init__(self,
                 window_size: int,
                 n_prediction_steps: int,
                 lagged_values: list[int],
                 sample_interval: int,
                 target_scaler: TargetScaler,
                 device: torch.device):
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps
        self.lagged_values = lagged_values
        self.sample_interval = sample_interval
        self.target_scaler = target_scaler
        self.cached_lag_mask_encoder = None
        self.cached_lag_mask_decoder = None
        self.device = device

    def preprocessing(self, X: dict):
        past_targets = X['past_targets'].float()
        past_features = X['past_features']
        if past_features is not None:
            past_features = past_features.float()
        past_observed_targets = X['past_observed_targets']
        future_features = X['future_features']
        if future_features is not None:
            future_features = future_features.float()

        if self.window_size < past_targets.shape[1]:
            past_targets = past_targets.to(self.device)
            past_observed_targets = past_observed_targets.to(self.device)
            past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler.transform(
                past_targets[:, -self.window_size:],
                past_observed_targets[:, -self.window_size:]
            )
            past_targets[:, :-self.window_size] = torch.where(
                past_observed_targets[:, :-self.window_size],
                scale_value(past_targets[:, :-self.window_size], loc, scale, device=self.device),
                past_targets[:, :-self.window_size])
        else:
            past_targets, _, loc, scale = self.target_scaler.transform(
                past_targets.to(self.device),
                past_observed_targets.to(self.device)
            )
        truncated_past_targets, self.cached_lag_mask_encoder = get_lagged_subsequences(past_targets,
                                                                                       self.window_size,
                                                                                       self.lagged_values,
                                                                                       self.cached_lag_mask_encoder)

        if past_features is not None:
            if self.window_size <= past_features.shape[1]:
                past_features = past_features[:, -self.window_size:]
            elif self.lagged_values:
                past_features = pad_tensor(past_features, self.window_size)

        if past_features is not None:
            past_features = past_features.to(device=self.device)
            x_past = torch.cat([truncated_past_targets, past_features], dim=-1)
        else:
            x_past = truncated_past_targets

        truncated_decoder_targets, self.cached_lag_mask_decoder = get_lagged_subsequences(past_targets,
                                                                                          self.n_prediction_steps,
                                                                                          self.lagged_values,
                                                                                          self.cached_lag_mask_decoder)
        """
        if future_features is not None:
            future_features = future_features.to(self.device)
            x_future = torch.cat(
                [future_features, truncated_decoder_targets], dim=-1
            )
        else:
            x_future = truncated_decoder_targets
        """
        x_future = future_features.to(self.device)

        return x_past, x_future, (loc, scale)