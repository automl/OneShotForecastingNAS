from abc import abstractmethod
from typing import Type

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta, Distribution, Gamma, Normal, Poisson, StudentT


class QuantileHead(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_output: int,
                 quantiles: list[float] | None = None,
                 val_out_idx: int | None = None
                 ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
            val_out_idx = 1
        else:
            assert val_out_idx is not None

        self.val_out_idx = val_out_idx
        self.nets = nn.ModuleList([nn.Linear(d_model, d_output) for _ in range(len(quantiles))])
        self.quantiles = quantiles

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [net(x) for net in self.nets]

    def loss(self, targets: torch.Tensor, predictions: list[torch.Tensor]):
        assert len(predictions) == len(self.quantiles)
        losses_all = []
        for q, y_pred in zip(self.quantiles, predictions):
            diff = targets - y_pred

            loss_q = torch.max(q * diff, (q - 1) * diff)
            losses_all.append(loss_q.unsqueeze(-1))

        losses_all = torch.mean(torch.concat(losses_all, dim=-1), dim=-1)

        return torch.mean(losses_all)

    def get_inference_pred(self, predictions: list[torch.Tensor]):
        return predictions[self.val_out_idx]


class MSEOutput(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_output: int,
                 **kwargs
                 ):
        super().__init__()
        self.net = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, targets: torch.Tensor, predictions: torch.Tensor):
        return torch.mean((targets - predictions) ** 2)

    def get_inference_pred(self, predictions: torch.Tensor):
        return predictions


class MAEOutput(MSEOutput):
    def loss(self, targets: torch.Tensor, predictions: torch.Tensor):
        return torch.mean(torch.abs(targets - predictions))

    def get_inference_pred(self, predictions: torch.Tensor):
        return predictions


class DistProjectionLayer(nn.Module):
    """
    A projection layer that project features to a torch distribution
    """

    value_in_support = 0.0

    # https://github.com/automl/Auto-PyTorch/blob/master/autoPyTorch/pipeline/components/setup/network_head/forecasting_network_head/distribution.py

    def __init__(
        self,
        d_model: int,
        d_output: int,
        inference_num_samples: int = 100,
        inference_samples_agg: str = 'median',
    ):
        super().__init__()

        self.proj = nn.ModuleList(
            [nn.Linear(d_model, dim * d_output) for dim in self.arg_dims.values()]
        )
        self.inference_num_samples = inference_num_samples
        self.inference_samples_agg = inference_samples_agg

    def forward(self, x: torch.Tensor) -> torch.distributions:
        """
        get a target distribution
        Args:
            x: input tensor ([batch_size, in_features]):
                input tensor, acquired by the base header, have the shape [batch_size, in_features]

        Returns:
            dist: torch.distributions ([batch_size, n_prediction_steps, output_shape]):
                an output torch distribution with shape (batch_size, n_prediction_steps, output_shape)
        """
        params_unbounded = [proj(x) for proj in self.proj]
        return self.dist_cls(*self.domain_map(*params_unbounded))

    @property
    @abstractmethod
    def arg_dims(self) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def domain_map(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_cls(self) -> Type[Distribution]:
        raise NotImplementedError

    def loss(self, targets: torch.Tensor, predictions: Distribution):
        return - predictions.log_prob(targets).mean()

    def get_inference_pred(self, predictions: Distribution):
        samples = predictions.rsample((self.inference_num_samples, ))
        if self.inference_samples_agg == 'mean':
            return torch.mean(samples, 0)
        elif self.inference_samples_agg == 'median':
            return torch.median(samples, 0)[0]
        else:
            raise NotImplementedError(f"Unknown inference_samples_agg {self.inference_samples_agg}")


class NormalOutput(DistProjectionLayer):
    @property
    def arg_dims(self) -> dict[str, int]:
        return {"loc": 1, "scale": 1}

    def domain_map(self, loc: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        scale = F.softplus(scale) + 1e-10
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Normal  # type: ignore[no-any-return]


class StudentTOutput(DistProjectionLayer):
    @property
    def arg_dims(self) -> dict[str, int]:
        return {"df": 1, "loc": 1, "scale": 1}

    def domain_map(  # type: ignore[override]
        self, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale) + 1e-10
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return StudentT   # type: ignore[no-any-return]


class BetaOutput(DistProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) -> dict[str, int]:
        return {"concentration1": 1, "concentration0": 1}

    def domain_map(  # type: ignore[override]
        self, concentration1: torch.Tensor, concentration0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(-1), concentration0.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        # TODO consider constraints on Beta!!!
        return Beta   # type: ignore[no-any-return]



class GammaOutput(DistProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) -> dict[str, int]:
        return {"concentration": 1, "rate": 1}

    def domain_map(  # type: ignore[override]
        self, concentration: torch.Tensor, rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO we need to adapt epsilon value given the datatype of this module
        epsilon = 1e-10
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(-1), rate.squeeze(-1)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Gamma  # type: ignore[no-any-return]


class PoissonOutput(DistProjectionLayer):
    @property
    def arg_dims(self) -> dict[str, int]:
        return {"rate": 1}

    def domain_map(self, rate: torch.Tensor) -> tuple[torch.Tensor]:  # type: ignore[override]
        rate_pos = F.softplus(rate).clone()
        return (rate_pos.squeeze(-1),)

    @property
    def dist_cls(self) -> Type[Distribution]:
        return Poisson  # type: ignore[no-any-return]
