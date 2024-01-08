from enum import Enum

import torch
import omegaconf


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
