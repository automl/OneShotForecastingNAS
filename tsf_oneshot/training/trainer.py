import os
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.contrib import tzip

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import get_lagged_subsequences

from tsf_oneshot.networks.architect import Architect
from tsf_oneshot.networks.network_controller import (
    ForecastingDARTSNetworkController,
    ForecastingGDASNetworkController,
    AbstractForecastingNetworkController
)

import wandb

from torch import nn
import torch
import numpy as np
from tsf_oneshot.training.training_utils import scale_value, rescale_output


def save_images(batch_idx, var_idx, kwargs):
    train_X = kwargs['train_X']
    target_train = kwargs['target_train']
    prediction_train = kwargs['prediction_train']
    val_X = kwargs['val_X']
    target_val = kwargs['target_val']
    prediction_val = kwargs['prediction_val']

    past = train_X['past_targets'][batch_idx, :, var_idx].cpu().detach().numpy()
    future = target_train[batch_idx, :, var_idx].cpu().detach().numpy()

    prediction = prediction_train[0][1][batch_idx, :, var_idx].cpu().detach().numpy()
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(past, label='Past')
    ax[0].plot(np.arange(len(past), len(past) + len(prediction)),
             future, label='GroundTRUTH')
    ax[0].plot(np.arange(len(past), len(past) + len(prediction)),
             prediction, label='Prediction')
    ax[0].legend()
    ax[0].set_title('Train')
    i = 0
    for _ in range(10000):
        if Path(f'train_{i}_batch_{batch_idx}_var_{var_idx}.png').exists():
            i += 1
        else:
            break

    past = val_X['past_targets'][batch_idx, :, var_idx].cpu().detach().numpy()
    future = target_val[batch_idx, :, var_idx].cpu().detach().numpy()

    prediction = prediction_val[0][1][batch_idx, :, var_idx].cpu().detach().numpy()

    ax[1].plot(past, label='Past')
    ax[1].plot(np.arange(len(past), len(past) + len(prediction)),
             future, label='GroundTRUTH')
    ax[1].plot(np.arange(len(past), len(past) + len(prediction)),
             prediction, label='Prediction')
    ax[1].legend()
    ax[1].set_title('val')
    fig.savefig(f'train_val{i}_batch_{batch_idx}_var_{var_idx}.png')


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


class ForecastingTrainer:
    def __init__(self,
                 model: AbstractForecastingNetworkController,
                 w_optimizer: torch.optim.Optimizer,
                 a_optimizer: torch.optim.Optimizer,
                 lr_scheduler_w: torch.optim.lr_scheduler.LRScheduler | None,
                 lr_scheduler_a: torch.optim.lr_scheduler.LRScheduler | None,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 window_size: int,
                 n_prediction_steps: int,
                 lagged_values: list[int],
                 target_scaler: TargetScaler,
                 grad_clip: float = 0,
                 device: torch.device = torch.device('cuda'),
                 amp_enable: bool = False,
                 ):

        self.model = model.to(device)
        self.w_optimizer = w_optimizer
        self.a_optimizer = a_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.target_scaler = target_scaler

        self.device = device

        self.lagged_values = lagged_values
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps

        self.cached_lag_mask_encoder = None
        self.cached_lag_mask_decoder = None

        self.lr_scheduler_w = lr_scheduler_w
        self.lr_scheduler_a = lr_scheduler_a

        self.grad_clip = grad_clip
        self.amp_enable = amp_enable
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

        self.backcast_loss_ration = model.backcast_loss_ration
        self.forecast_only = model.forecast_only

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

    def train_epoch(self, epoch: int):
        if self.lr_scheduler_w is not None:
            self.lr_scheduler_w.step()
        if self.lr_scheduler_a is not None:
            self.lr_scheduler_a.step()

        for (train_X, train_y), (val_X, val_y) in tzip(self.train_loader, self.val_loader):
            # update model weights
            torch.cuda.empty_cache()
            w_loss, _ = self.update_weights(train_X, train_y)

            torch.cuda.empty_cache()
            self.update_alphas(val_X, val_y)

            """
            w_loss, prediction_train = self.update_weights(train_X, train_y)
            a_loss, prediction_val = self.update_alphas(val_X, val_y)

            from functools import partial

            kwargs = {
                'train_X': train_X,
                'target_train': train_y['future_targets'].float(),
                'prediction_train': prediction_train,
                'val_X': val_X,
                'target_val': val_y['future_targets'].float(),
                'prediction_val': prediction_val,
            }
            func = partial(save_images, kwargs=kwargs)
            import pdb
            pdb.set_trace()
            #"""

        return w_loss.detach().cpu()

    def update_weights(self, train_X, train_y):
        x_past_train, x_future_train, scale_value_train = self.preprocessing(train_X)
        target_train = train_y['future_targets'].float().to(self.device)

        with torch.cuda.amp.autocast(enabled=self.amp_enable):
            prediction_train, w_dag_train = self.model(x_past_train, x_future_train, return_w_head=True)
            prediction = rescale_output(prediction_train, *scale_value_train, device=self.device)
            if self.forecast_only:
                loss_all = self.model.get_individual_training_loss(
                    target_train, prediction,
                )
            else:
                backcast, forecast = prediction
                target_train_backcast = train_X['past_targets'].float()[:, -self.window_size:, :].to(self.device)
                loss_backcast = self.model.get_individual_training_loss(
                    target_train_backcast, backcast
                )
                loss_forecast = self.model.get_individual_training_loss(
                    target_train, forecast,
                )
                loss_all = [l_b * self.backcast_loss_ration + l_f for l_b, l_f in zip(loss_backcast, loss_forecast)]

            if isinstance(self.model, ForecastingDARTSNetworkController):
                w_loss = sum([w * loss for w, loss in zip(w_dag_train[0], loss_all)])
            elif isinstance(self.model, ForecastingGDASNetworkController):
                hardwts, index = w_dag_train
                hardwts = hardwts[0]
                index = index[0]
                w_loss = sum(
                    hardwts[_ie] * loss if _ie == index else hardwts[_ie]
                    for _ie, loss in enumerate(loss_all)
                )
            else:
                raise NotImplementedError

        self.scaler.scale(w_loss).backward()
        wandb.log({'weights training loss': w_loss})
        if self.forecast_only:
            for i, loss in enumerate(loss_all):
                wandb.log({f'training loss {i}': loss})
        else:
            for i, (l_b, l_f) in enumerate(zip(loss_backcast, loss_forecast)):
                wandb.log({f'training loss backcast {i}': l_b})
                wandb.log({f'training loss forecast {i}': l_f})

        if self.grad_clip > 0:
            self.scaler.unscale_(self.w_optimizer)
            gradient_norm = self.model.grad_norm_weights()
            wandb.log({'gradient_norm_weights': gradient_norm})
            torch.nn.utils.clip_grad_norm_(self.model.weights(), self.grad_clip)
        else:
            self.scaler.unscale_(self.w_optimizer)
            gradient_norm = self.model.grad_norm_weights()
            wandb.log({'gradient_norm_weights': gradient_norm})

        self.scaler.step(self.w_optimizer)
        self.scaler.update()

        self.w_optimizer.zero_grad()

        return w_loss, prediction

    def update_alphas(self, val_X, val_y):
        x_past_val, x_future_val, scale_value_val = self.preprocessing(val_X)
        target_val = val_y['future_targets'].float().to(self.device)

        with torch.cuda.amp.autocast(enabled=self.amp_enable):

            prediction_val, w_dag_val = self.model(x_past_val, x_future_val, return_w_head=True)
            prediction = rescale_output(prediction_val, *scale_value_val, device=self.device)

            if self.forecast_only:
                a_loss = self.model.get_validation_loss(
                    target_val, prediction, w_dag_val
                )
            else:
                backcast, forecast = prediction
                target_val_backcast = val_X['past_targets'].float()[:, -self.window_size:, :].to(self.device)
                loss_backcast = self.model.get_validation_loss(
                    target_val_backcast, backcast, w_dag_val
                )
                loss_forecast = self.model.get_validation_loss(
                    target_val, forecast, w_dag_val
                )
                a_loss = loss_backcast * self.backcast_loss_ration + loss_forecast

        self.scaler.scale(a_loss).backward()

        wandb.log({'alphas training loss': a_loss})

        if self.grad_clip > 0:
            self.scaler.unscale_(self.a_optimizer)
            gradient_norm = self.model.grad_norm_alphas()
            wandb.log({'gradient_norm_alphas': gradient_norm})
            torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.grad_clip)
        else:
            self.scaler.unscale_(self.a_optimizer)
            gradient_norm = self.model.grad_norm_alphas()
            wandb.log({'gradient_norm_alphas': gradient_norm})

        self.scaler.step(self.a_optimizer)
        self.scaler.update()
        self.a_optimizer.zero_grad()

        return a_loss, prediction

    def save(self, save_path: Path, epoch: int):
        if not save_path.exists():
            os.makedirs(save_path)
        self.model.save(save_path / 'Model')

        save_dict = {
            'w_optimizer': self.w_optimizer.state_dict(),
            'a_optimizer': self.a_optimizer.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler_w is not None:
            save_dict['lr_scheduler_w'] = self.lr_scheduler_w.state_dict()
        if self.lr_scheduler_a is not None:
            save_dict['lr_scheduler_a'] = self.lr_scheduler_a.state_dict()

        torch.save(
            save_dict,
            save_path / 'trainer_info.pth'
        )

    @staticmethod
    def load(resume_path: Path,
             model: AbstractForecastingNetworkController,
             w_optimizer: torch.optim.Optimizer, a_optimizer: torch.optim.Optimizer,
             lr_scheduler_w: torch.optim.lr_scheduler.LRScheduler | None,
             lr_scheduler_a: torch.optim.lr_scheduler.LRScheduler | None = None
             ):
        model.load(resume_path / 'Model', model=model)

        trainer_info = torch.load(resume_path / 'trainer_info.pth')
        w_optimizer.load_state_dict(trainer_info['w_optimizer'])
        a_optimizer.load_state_dict(trainer_info['a_optimizer'])

        if lr_scheduler_w is not None:
            lr_scheduler_w.load_state_dict(trainer_info['lr_scheduler_w'])
        if lr_scheduler_a is not None:
            lr_scheduler_a.load_state_dict(trainer_info['lr_scheduler_a'])

        return trainer_info.get('epoch', 0)


class ForecastingDARTSSecondOrderTrainer(ForecastingTrainer):
    def __init__(self,
                 model: ForecastingDARTSNetworkController,
                 architect: Architect,
                 lr_init:float | None=None,
                 **kwargs,
                 ):
        super(ForecastingDARTSSecondOrderTrainer, self).__init__(model=model, **kwargs)
        self.architect = architect
        if lr_init is None and kwargs['lr_scheduler_w'] is None:
            # TODO check if amp works on this type of architect
            raise ValueError("either lr_scheduler_w or lr_init must be given")
        self.lr_init = lr_init


    def train_epoch(self, epoch: int, update_alpha:bool=False):
        if self.lr_scheduler_w is not None:
            self.lr_scheduler_w.step()
            lr = self.lr_scheduler_w.get_lr()[0]
        else:
            lr = self.lr_init

        if self.lr_scheduler_a is not None:
            self.lr_scheduler_a.step()

        for step, ((train_X, train_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.val_loader)):
            x_past_train, x_future_train, scale_value_train = self.preprocessing(train_X)
            x_past_val, x_future_val, scale_value_val = self.preprocessing(val_X)

            train_y = train_y.float().to(self.device)
            val_y = val_y.float().to(self.device)

            self.a_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                self.architect.unrolled_backward(x_past_train, x_future_train, train_y, scale_value_train,
                                                 x_past_val, x_future_val, val_y, scale_value_val,
                                                 lr, self.w_optimizer,self.amp_enable, self.scaler,
                                                 )

            if self.grad_clip > 0:
                self.scaler.unscale_(self.a_optimizer)
                gradient_norm = self.model.grad_norm_alphas()
                wandb.log({'gradient_norm_alphas': gradient_norm})
                torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.grad_clip)
            else:
                self.scaler.unscale_(self.a_optimizer)
                gradient_norm = self.model.grad_norm_alphas()
                wandb.log({'gradient_norm_alphas': gradient_norm})

            self.scaler.step(self.a_optimizer)
            self.scaler.update()

            # optimize the weights
            # update model weights
            self.w_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                prediction_train, w_dag_train = self.model(x_past_train, x_future_train, return_w_head=True)

                w_loss = self.model.get_training_loss(train_y,
                                                      rescale_output(prediction_train, *scale_value_train, device=self.device),
                                                      w_dag_train
                                                      )
            self.scaler.scale(w_loss).backward()
            wandb.log({'weights training loss': w_loss})

            if self.grad_clip > 0:
                self.scaler.unscale_(self.w_optimizer)
                gradient_norm = self.model.grad_norm_weights()
                wandb.log({'gradient_norm_weights': gradient_norm})
                torch.nn.utils.clip_grad_norm_(self.model.weights(), self.grad_clip)
            else:
                self.scaler.unscale_(self.w_optimizer)
                gradient_norm = self.model.grad_norm_weights()
                wandb.log({'gradient_norm_weights': gradient_norm})

            self.scaler.step(self.w_optimizer)
            self.scaler.update()

