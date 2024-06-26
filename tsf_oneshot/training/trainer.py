import copy
import os
import pathlib
from pathlib import Path

import omegaconf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tzip

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.utils import TargetScaler
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import get_lagged_subsequences

from tsf_oneshot.networks.architect import Architect
from tsf_oneshot.networks.network_controller import (
    ForecastingDARTSNetworkController,
    ForecastingGDASNetworkController,
    AbstractForecastingNetworkController,
    N_RESERVED_EDGES_PER_NODE,
)

import wandb

from torch import nn
import torch
import numpy as np
from tsf_oneshot.training.training_utils import scale_value, rescale_output
from tsf_oneshot.training.utils import get_optimizer, LR_SCHEDULER_TYPE


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
                 val_eval_loader: torch.utils.data.DataLoader | None,
                 proj_intv: int,
                 proj_intv_nodes: int,
                 window_size: int,
                 n_prediction_steps: int,
                 sample_interval: int,
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

        self.val_eval_loader = val_eval_loader
        self.proj_intv = proj_intv
        self.proj_intv_nodes = proj_intv_nodes

        self.target_scaler = target_scaler

        self.device = device

        self.lagged_values = lagged_values
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps
        self.sample_interval = sample_interval

        self.target_indices = torch.arange(0, n_prediction_steps, sample_interval)
        self.data_indices = torch.arange(0, window_size, sample_interval)

        self.cached_lag_mask_encoder = None
        self.cached_lag_mask_decoder = None

        self.lr_scheduler_w = lr_scheduler_w
        self.lr_scheduler_a = lr_scheduler_a

        if isinstance(self.lr_scheduler_w, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler_w_type = LR_SCHEDULER_TYPE.batch
        else:
            self.lr_scheduler_w_type = LR_SCHEDULER_TYPE.epoch

        if isinstance(self.lr_scheduler_a, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler_a_type = LR_SCHEDULER_TYPE.batch
        else:
            self.lr_scheduler_a_type = LR_SCHEDULER_TYPE.epoch

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

    def train_epoch(self, epoch: int, update_alphas:bool=True):
        self.model.train()

        for (train_X, train_y), (val_X, val_y) in tzip(self.train_loader, self.val_loader):
            # update model weights
            torch.cuda.empty_cache()
            w_loss, _ = self.update_weights(train_X, train_y)

            if update_alphas:
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
        if self.lr_scheduler_w is not None and self.lr_scheduler_w_type == LR_SCHEDULER_TYPE.epoch:
            self.lr_scheduler_w.step()
        if self.lr_scheduler_a is not None and self.lr_scheduler_a_type == LR_SCHEDULER_TYPE.epoch:
            self.lr_scheduler_a.step()

        return w_loss.detach().cpu()

    def update_weights(self, train_X, train_y):
        x_past_train, x_future_train, scale_value_train = self.preprocessing(train_X)
        target_train = train_y['future_targets'].float().to(self.device)

        shift = np.random.choice(self.sample_interval)
        #x_past_train_ = x_past_train.clone()
        #x_future_train_ = x_future_train.clone()
        #target_train_ = target_train.clone()

        x_past_train = x_past_train[:, self.data_indices + shift]
        x_future_train = x_future_train[:, self.target_indices + shift]
        target_train = target_train[:, self.target_indices + shift]

        self.model.train()


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
        self.a_optimizer.zero_grad()

        if self.lr_scheduler_w is not None and self.lr_scheduler_w_type == LR_SCHEDULER_TYPE.epoch:
            self.lr_scheduler_w.step()

        return w_loss, prediction

    def update_alphas(self, val_X, val_y):
        x_past_val, x_future_val, scale_value_val = self.preprocessing(val_X)
        target_val = val_y['future_targets'].float().to(self.device)

        shift = np.random.choice(self.sample_interval)
        x_past_val_ = x_past_val.clone()
        x_future_val_ = x_future_val.clone()
        target_val_ = target_val.clone()

        x_past_val = x_past_val_[:, self.data_indices + shift]
        x_future_val = x_future_val_[:, self.target_indices + shift]
        target_val = target_val_[:, self.target_indices + shift]

        self.model.on_validate()

        with torch.cuda.amp.autocast(enabled=self.amp_enable):

            prediction_val, w_dag_val = self.model(x_past_val, x_future_val, return_w_head=True)
            prediction = rescale_output(prediction_val, *scale_value_val, device=self.device)

            if self.forecast_only:
                a_loss = self.model.get_validation_loss(
                    target_val, prediction, w_dag_val
                )
            else:
                backcast, forecast = prediction
                a_loss = self.model.get_validation_loss(
                    target_val, forecast, w_dag_val
                )

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
        self.w_optimizer.zero_grad()
        self.a_optimizer.zero_grad()

        if self.lr_scheduler_a is not None and self.lr_scheduler_a_type == LR_SCHEDULER_TYPE.epoch:
            self.lr_scheduler_a.step()

        return a_loss, prediction

    def evaluate(self, test_loader, epoch, loader_type: str = 'test'):
        mse_losses = []
        mae_losses = []
        num_data_points = 0

        for (test_X, test_y) in tqdm(test_loader):
            x_past_test, x_future_test, scale_value_test = self.preprocessing(test_X)
            target_test = test_y['future_targets'].float()
            n_data = len(target_test)

            x_past_test_ = x_past_test.clone()
            x_future_test_ = x_future_test.clone()

            pred_test = torch.zeros_like(target_test)

            self.model.eval()
            for shift in range(self.sample_interval):
                x_past_test = x_past_test_[:, self.data_indices + shift]
                x_future_test = x_future_test_[:, self.target_indices + shift]
                with torch.no_grad():
                    prediction_test, w_dag_test = self.model(x_past_test, x_future_test, return_w_head=True)
                    if not self.forecast_only:
                        prediction_test = prediction_test[-1]
                    prediction_test = self.model.get_inference_prediction(prediction_test, w_dag_test)
                    prediction_test = rescale_output(prediction_test, *scale_value_test, device=self.device).cpu()

                pred_test[:, self.target_indices + shift] = prediction_test

            diff = (pred_test - target_test)

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

        if lr_scheduler_w is not None and 'lr_scheduler_w' in trainer_info:
            lr_scheduler_w.load_state_dict(trainer_info['lr_scheduler_w'])
        if lr_scheduler_a is not None and 'lr_scheduler_a' in trainer_info:
            lr_scheduler_a.load_state_dict(trainer_info['lr_scheduler_a'])

        return trainer_info.get('epoch', 0)

    def pt_project(self, cfg: omegaconf.DictConfig, proj_path:pathlib.Path, reset_optimizer: bool=True):
        if reset_optimizer:
            # original from https://github.com/ruocwang/darts-pt
            self.lr_scheduler_w = None
            self.lr_scheduler_a = None

            del self.w_optimizer
            # reset w_optimizer
            cfg_w_optimizer = copy.copy(cfg.w_optimizer)
            cfg_w_optimizer.lr = cfg_w_optimizer.lr / 10
            self.w_optimizer = get_optimizer(
                cfg_optimizer=cfg_w_optimizer,
                optim_groups=self.model.get_weight_optimizer_parameters(cfg_w_optimizer.weight_decay),
                wd_in_p_groups=True
            )
        num_edges = sum(self.model.candidate_flag_ops)
        tune_epochs = self.proj_intv * (num_edges - 1) + 1
        # prune the objects
        for epoch in range(tune_epochs):
            if epoch % self.proj_intv == 0 or epoch == tune_epochs - 1:
                if epoch == tune_epochs - 1:
                    # We make decision on this only in the last itartion!
                    selected_eid = num_edges - 1
                else:
                    candidate_edges = np.nonzero(self.model.candidate_flag_ops[:-1])[0]
                    if len(candidate_edges) == 0:
                        continue
                    selected_eid = np.random.choice(candidate_edges)
                selected_eid_raw = selected_eid
                for n_edge, mask_name in zip(self.model.len_all_arch_p, self.model.all_mask_names):
                    if 0 <= selected_eid < n_edge:
                        # in this case we have selected the correct
                        mask: torch.Tensor = getattr(self.model, mask_name)
                        n_ops = mask.shape[-1]
                        best_opid = 0
                        if n_ops <= 1:
                            continue
                        loss_best = -np.inf
                        for op_id in range(n_ops):
                            # we remove the corresponding operations.
                            mask[selected_eid] = 0
                            mask[selected_eid, op_id] = -torch.inf

                            loss = self.evaluate(self.val_eval_loader, epoch=epoch* 100 + op_id, loader_type='proj')['MSE loss']
                            if np.isnan(loss):
                                return False

                            if loss > loss_best:
                                loss_best = loss
                                best_opid = op_id
                        # we have evaluated all the operations, now we can safely return
                        self.model.candidate_flag_ops[selected_eid_raw] = False
                        mask[selected_eid] = -torch.inf
                        mask[selected_eid, best_opid] = 0
                        break
                    else:
                        selected_eid -= n_edge
            w_loss = self.train_epoch(epoch)
            os.makedirs(proj_path, exist_ok=True)
            if not torch.isnan(w_loss):
                self.save(proj_path, epoch)
            else:
                return False
        return True

    def pt_project_topology(self):
        # original from https://github.com/ruocwang/darts-pt
        num_nodes = sum(self.model.candidate_flag_nodes)

        tune_epochs = self.proj_intv_nodes * (num_nodes - 1)

        # topology selection
        def set_value_for_mask(select_e_ids, new_value):
            raw_values = []
            for i, selected_eid in enumerate(select_e_ids):
                # A function used for mask or unmask an edge
                for n_edge, mask_name in zip(self.model.len_all_arch_p, self.model.all_mask_names):
                    if 0 <= selected_eid < n_edge:
                        # in this case we have selected the correct
                        mask: nn.Parameter = getattr(self.model, mask_name)
                        # mask out the edge
                        mask_raw = mask.data[selected_eid].clone()
                        raw_values.append(mask_raw)
                        if isinstance(new_value, list):
                            mask.data[selected_eid].copy_(new_value[i])
                        else:
                            mask.data[selected_eid].copy_(new_value)
                        selected_eid -= n_edge
                    else:
                        selected_eid -= n_edge
            return raw_values

        for epoch in range(tune_epochs):
            if epoch % self.proj_intv_nodes == 0 or epoch == tune_epochs - 1:
                candidate_nodes = np.nonzero(self.model.candidate_flag_nodes)[0]
                selected_nid = np.random.choice(candidate_nodes)
                selected_nid_raw = selected_nid

                # select all candidate nodes:

                all_edges = set(np.arange(selected_nid))
                while(len(all_edges) > N_RESERVED_EDGES_PER_NODE):
                    print(f'start to pure the node {selected_nid}, remaining edges are {all_edges}')
                    loss_worst = np.inf
                    # we need to remove the edge that is the most unimportant
                    worst_eid = None
                    edges_to_remove = []
                    for j in all_edges:
                        node = f'{selected_nid}<-{j}'
                        if node in self.model.edges2index:
                            e_ids = self.model.edges2index[node]
                            # mask out all the related nodes
                            mask_raw_value = set_value_for_mask(e_ids, -torch.inf)
                            loss = self.evaluate(self.val_eval_loader, epoch=j, loader_type='proj')['MSE loss']
                            if loss < loss_worst:
                                loss_worst = loss
                                worst_eid = j
                            # return all the nodes
                            set_value_for_mask(e_ids, mask_raw_value)
                        else:
                            edges_to_remove.append(j)
                    for edge in edges_to_remove:
                        all_edges.remove(edge)

                    # mask out all the unrelated edges.
                    print(f'remove edge {worst_eid}')
                    all_edges.remove(worst_eid)
                    node = f'{selected_nid}<-{worst_eid}'
                    e_ids = self.model.edges2index[node]
                    set_value_for_mask(e_ids, -torch.inf)

                self.model.candidate_flag_nodes[selected_nid_raw] = False
            # since all the edges only contain one value, there is no need to update alphas
            self.train_epoch(epoch, update_alphas=False)


class ForecastingDARTSSecondOrderTrainer(ForecastingTrainer):
    def __init__(self,
                 model: ForecastingDARTSNetworkController,
                 architect: Architect,
                 lr_init: float | None = None,
                 **kwargs,
                 ):
        super(ForecastingDARTSSecondOrderTrainer, self).__init__(model=model, **kwargs)
        self.architect = architect
        if lr_init is None and kwargs['lr_scheduler_w'] is None:
            # TODO check if amp works on this type of architect
            raise ValueError("either lr_scheduler_w or lr_init must be given")
        self.lr_init = lr_init

    def train_epoch(self, epoch: int, update_alpha: bool = False):
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

            train_y = train_y[:, self.target_indices].float().to(self.device)
            val_y = val_y[:, self.tFarget_indices].float().to(self.device)

            self.a_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                self.architect.unrolled_backward(x_past_train, x_future_train, train_y, scale_value_train,
                                                 x_past_val, x_future_val, val_y, scale_value_val,
                                                 lr, self.w_optimizer, self.amp_enable, self.scaler,
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
                                                      rescale_output(prediction_train, *scale_value_train,
                                                                     device=self.device),
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
