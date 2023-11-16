""" Architect controls architecture of cell by computing gradients of alphas """
""" Used for DARTS model Originally Implemented from https://github.com/khanrc/pt.darts/blob/master/architect.py"""
import copy
import torch

from tsf_oneshot.networks.network_controller import AbstractForecastingNetworkController
from tsf_oneshot.training_utils import rescale_output


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net: AbstractForecastingNetworkController, w_momentum: float, w_weight_decay: float):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, x_past: torch.Tensor, x_future: torch.Tensor, targets: torch.Tensor, scale_info, xi, w_optim,
                     amp_enable: bool, amp_scaler:  torch.cuda.amp.GradScaler,):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc los
        with torch.cuda.amp.autocast(enabled=amp_enable):
            predictions, w_dag = self.net(x_past, x_future, return_w_head=True)
            loss = self.net.get_training_loss(targets=targets,
                                              predictions=rescale_output(predictions, *scale_info,
                                                                         device=targets.device), w_dag=w_dag) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(amp_scaler.scale(loss), self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        amp_scaler.unscale_(w_optim)
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.arch_parameters()):
                va.copy_(a)

    def unrolled_backward(self,
                          x_past_train: torch.Tensor, x_future_train: torch.Tensor, targets_train: torch.Tensor, scale_info_train,
                          x_past_val: torch.Tensor, x_future_val: torch.Tensor, targets_val: torch.Tensor, scale_info_val,
                          xi, w_optim,
                          amp_enable: bool, amp_scaler:  torch.cuda.amp.GradScaler,):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(x_past_train, x_future_train, targets_train, scale_info_train,xi, w_optim, amp_enable, amp_scaler)
        # TODO try using amp on all losses and grad functions!
        with torch.cuda.amp.autocast(enabled=amp_enable):
            predictions, w_dag = self.net(x_past_val, x_future_val, return_w_head=True)

            loss = self.net.get_validation_loss(targets=targets_val,
                                                predictions=rescale_output(predictions, *scale_info_val,
                                                                           device=targets_val.device),
                                                w_dag=w_dag) # L_trn(w)

        # compute gradient
        v_alphas = tuple(self.v_net.arch_parameters())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(amp_scaler.scale(loss), v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, x_past_train, x_future_train, targets_train, scale_info_train, amp_enable,
                                       amp_scaler)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, x_past, x_future, targets, scale_info, amp_enable, amp_scaler):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        with torch.cuda.amp.autocast(enabled=amp_enable):
            predictions, w_dag = self.net(x_past, x_future, return_w_head=True)

            loss = self.net.get_training_loss(targets=targets,
                                              predictions=rescale_output(predictions, *scale_info, device=targets.device),
                                              w_dag=w_dag)  # L_trn(w)
        dalpha_pos = torch.autograd.grad(amp_scaler.scale(loss), self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        with torch.cuda.amp.autocast(enabled=amp_enable):

            predictions, w_dag = self.net(x_past, x_future, return_w_head=True)
            loss = self.net.get_training_loss(targets=targets,
                                              predictions=rescale_output(predictions, *scale_info,device=targets.device),
                                              w_dag=w_dag)
        dalpha_neg = torch.autograd.grad(amp_scaler.scale(loss), self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian