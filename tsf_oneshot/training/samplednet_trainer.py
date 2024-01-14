import os
from pathlib import Path

from tqdm import tqdm

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import get_lagged_subsequences

import wandb

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from tsf_oneshot.networks.sampled_net import SampledNet
from tsf_oneshot.training.training_utils import scale_value, rescale_output
from tsf_oneshot.training.trainer import pad_tensor
from tsf_oneshot.training.utils import LR_SCHEDULER_TYPE


class EarlyStopping:
    def __init__(self, tolerance: int = 10):
        self.tolerance = tolerance
        self.no_improvement = 0
        self.best_epoch = 0
        self.best_val_loss_mse = np.inf
        self.best_val_loss = {}
        self.best_test_loss = {}

        self.val_ht = []
        self.test_ht = []

    def __call__(self, val_loss: dict, test_loss:dict, n_epoch: int):
        val_loss_mse = val_loss['MSE loss']
        self.val_ht.append(val_loss)
        self.test_ht.append(test_loss)
        if val_loss_mse < self.best_val_loss_mse:
            self.best_val_loss_mse = val_loss_mse

            self.best_val_loss = val_loss
            self.best_test_loss = test_loss
            self.best_epoch = n_epoch

            self.no_improvement = 0
            return False
        else:
            self.no_improvement += 1
            if self.no_improvement >= self.tolerance:
                print(f'no improvement in {self.tolerance} epochs, early stops,'
                      f'best val loss is {self.best_val_loss},'
                      f'best test loss is {self.best_test_loss}')
                return True



def save_images(batch_idx, var_idx, kwargs):
    train_X = kwargs['train_X']
    target_train = kwargs['target_train']
    prediction_train = kwargs['prediction_train']
    val_X = kwargs['val_X']
    target_val = kwargs['target_val']
    prediction_val = kwargs['prediction_val']

    past = train_X['past_targets'][batch_idx, :, var_idx].cpu().detach().numpy()
    future = target_train[batch_idx, :, var_idx].cpu().detach().numpy()

    prediction = prediction_train[batch_idx, :, var_idx].cpu().detach().numpy()
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

    prediction = prediction_val[batch_idx, :, var_idx].cpu().detach().numpy()

    ax[1].plot(past, label='Past')
    ax[1].plot(np.arange(len(past), len(past) + len(prediction)),
             future, label='GroundTRUTH')
    ax[1].plot(np.arange(len(past), len(past) + len(prediction)),
             prediction, label='Prediction')
    ax[1].legend()
    ax[1].set_title('val')
    fig.savefig(f'train_val{i}_batch_{batch_idx}_var_{var_idx}.png')


class SampledForecastingNetTrainer:
    def __init__(self,
                 model: SampledNet,
                 w_optimizer: torch.optim.Optimizer,
                 lr_scheduler_w: torch.optim.lr_scheduler.LRScheduler | None,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
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

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.target_scaler = target_scaler

        self.device = device

        self.lagged_values = lagged_values
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps

        self.cached_lag_mask_encoder = None
        self.cached_lag_mask_decoder = None

        self.lr_scheduler_w = lr_scheduler_w

        if isinstance(self.lr_scheduler_w, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler_type = LR_SCHEDULER_TYPE.batch
        else:
            self.lr_scheduler_type = LR_SCHEDULER_TYPE.epoch

        self.grad_clip = grad_clip
        self.amp_enable = amp_enable
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

        self.forecast_only = model.forecast_only
        self.backcast_loss_ration = model.backcast_loss_ration

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

        # train model
        #"""
        self.model.train()
        for (train_X, train_y) in tqdm(self.train_loader):
            # update model weights

            torch.cuda.empty_cache()
            w_loss, _ = self.update_weights(train_X, train_y)
            torch.cuda.empty_cache()

        train_res = self.evaluate(self.train_loader, epoch, 'train')

        val_res = self.evaluate(self.val_loader, epoch, 'val')
        test_res = self.evaluate(self.test_loader, epoch, 'test')

        if self.lr_scheduler_w is not None and self.lr_scheduler_type == LR_SCHEDULER_TYPE.epoch:
            self.lr_scheduler_w.step()

        return val_res, test_res
        #"""

        #self.evaluate_with_plot()

    def evaluate(self, test_loader, epoch, loader_type: str = 'test'):
        mse_losses = []
        mae_losses = []
        num_data_points = 0
        self.model.eval()

        for (test_X, test_y) in tqdm(test_loader):
            x_past_test, x_future_test, scale_value_test = self.preprocessing(test_X)
            target_test = test_y['future_targets'].float()
            n_data = len(target_test)

            with torch.no_grad():
                prediction_test = self.model(x_past_test, x_future_test)
                prediction_test = self.model.get_inference_prediction(prediction_test)
                if not self.forecast_only:
                    prediction_test = prediction_test[-1]
                prediction_test = rescale_output(prediction_test, *scale_value_test, device=self.device).cpu()
                diff = (prediction_test - target_test)

            mse_losses.append(n_data * torch.mean(diff ** 2).double().item())
            mae_losses.append(n_data * torch.mean(torch.abs(diff)).item())
            num_data_points += n_data

        mean_mse_loses = sum(mse_losses) / num_data_points
        mean_mae_losses = sum(mae_losses) / num_data_points
        print(f'{loader_type} mse loss at epoch {epoch}: {mean_mse_loses}')
        print(f'{loader_type} mae loss at epochf {epoch}: {mean_mae_losses}')
        wandb.log({
            f'{loader_type} MSE loss': mean_mse_loses,
            f'{loader_type} MAE loss': mean_mae_losses
        })
        return {
            'MSE loss': mean_mse_loses,
            'MAE loss': mean_mae_losses
        }

    def evaluate_with_plot(self):
        self.model.eval()
        for (test_X, test_y) in tqdm(self.test_loader):
            x_past_test, x_future_test, scale_value_test = self.preprocessing(test_X)
            target_test = test_y['future_targets'].float()
            n_data = len(target_test)

            with torch.no_grad():
                prediction_test = self.model(x_past_test, x_future_test)
                prediction_test = self.model.get_inference_prediction(prediction_test)
                prediction_test = rescale_output(prediction_test, *scale_value_test, device=self.device).cpu()
                diff_test = (prediction_test - target_test)

            train_X, train_y = next(iter(self.train_loader))
            x_past_train, x_future_train, scale_value_train = self.preprocessing(train_X)
            target_train = train_y['future_targets'].float()

            with torch.no_grad():
                prediction_train = self.model(x_past_train, x_future_train)
                prediction_train = self.model.get_inference_prediction(prediction_train)
                prediction_train = rescale_output(prediction_train, *scale_value_train, device=self.device).cpu()
                diff_train = (target_train - prediction_train)

            from functools import partial

            kwargs = {
                'train_X': train_X,
                'target_train': train_y['future_targets'].float(),
                'prediction_train': prediction_train,
                'val_X': test_X,
                'target_val': test_y['future_targets'].float(),
                'prediction_val': prediction_test,
            }
            func = partial(save_images, kwargs=kwargs)

    def update_weights(self, train_X, train_y):
        x_past_train, x_future_train, scale_value_train = self.preprocessing(train_X)
        target_train = train_y['future_targets'].float().to(self.device)

        self.w_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.amp_enable):
            prediction_train = self.model(x_past_train, x_future_train, )
            prediction = rescale_output(prediction_train, *scale_value_train, device=self.device)
            if self.forecast_only:
                w_loss = self.model.get_training_loss(
                    target_train, prediction,
                )
            else:
                backcast, forecast = prediction
                target_train_backcast = train_X['past_targets'].float()[:, -self.window_size:, :].to(self.device)
                loss_backcast = self.model.get_training_loss(
                    target_train_backcast, backcast
                )
                loss_forecast = self.model.get_training_loss(
                    target_train, forecast,
                )
                w_loss = loss_backcast * self.backcast_loss_ration + loss_forecast


        self.scaler.scale(w_loss).backward()
        wandb.log({'weights training loss': w_loss})

        if self.grad_clip > 0:
            self.scaler.unscale_(self.w_optimizer)
            gradient_norm = self.model.grad_norm()
            wandb.log({'gradient_norm_weights': gradient_norm})
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            self.scaler.unscale_(self.w_optimizer)
            gradient_norm = self.model.grad_norm()
            wandb.log({'gradient_norm_weights': gradient_norm})
        self.scaler.step(self.w_optimizer)
        self.scaler.update()

        if self.lr_scheduler_w is not None and self.lr_scheduler_type == LR_SCHEDULER_TYPE.batch:
            self.lr_scheduler_w.step()

        return w_loss, prediction

    def save(self, save_path: Path, epoch: int):
        save_path = save_path / 'SampledNet'
        if not save_path.exists():
            os.makedirs(save_path)
        self.model.save(save_path / 'Model')

        save_dict = {
            'w_optimizer': self.w_optimizer.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler_w is not None:
            save_dict['lr_scheduler_w'] = self.lr_scheduler_w.state_dict()

        torch.save(
            save_dict,
            save_path / 'trainer_info.pth'
        )

    @staticmethod
    def load(resume_path: Path,
             model: SampledNet,
             w_optimizer: torch.optim.Optimizer,
             lr_scheduler_w: torch.optim.lr_scheduler.LRScheduler | None,
             ):
        resume_path = resume_path / 'SampledNet'
        model.load(resume_path / 'Model', model=model)

        trainer_info = torch.load(resume_path / 'trainer_info.pth')
        w_optimizer.load_state_dict(trainer_info['w_optimizer'])

        if lr_scheduler_w is not None:
            lr_scheduler_w.load_state_dict(trainer_info['lr_scheduler_w'])

        return trainer_info.get('epoch', 0)

