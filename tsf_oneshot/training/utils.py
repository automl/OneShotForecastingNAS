import torch
import omegaconf


def get_optimizer(cfg_optimizer: omegaconf.DictConfig,
                  optim_groups: list[torch.nn.Parameter],
                  wd_in_p_groups: bool = False) -> torch.optim.Optimizer:
    weight_decay = 0 if wd_in_p_groups else cfg_optimizer.weight_decay
    if cfg_optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=cfg_optimizer.lr,
            betas=(cfg_optimizer.beta1, cfg_optimizer.beta2),
            weight_decay=weight_decay
        )
    elif cfg_optimizer.type == 'sgd':
        optimizer = torch.optim.SGD(
            params=optim_groups,
            lr=cfg_optimizer.lr,
            momentum=cfg_optimizer.momentum,
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
        raise NotImplementedError
    return optimizer
