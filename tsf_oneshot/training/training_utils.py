from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import (TransformedDistribution_,
                                                                                    ALL_NET_OUTPUT)

from typing import Optional

import torch
from torch.distributions import AffineTransform


def scale_value(raw_value: torch.Tensor,
                loc: Optional[torch.Tensor],
                scale: Optional[torch.Tensor],
                device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    scale the outputs

    Args:
        raw_value (torch.Tensor):
            network head output
        loc (Optional[torch.Tensor]):
            scaling location value
        scale (Optional[torch.Tensor]):
            scaling scale value
        device (torch.device):
            which device the output is stored

    Return:
        torch.Tensor:
            scaled input value
    """
    if loc is not None or scale is not None:
        if loc is None:
            outputs = raw_value / scale.to(device)  # type: ignore[union-attr]
        elif scale is None:
            outputs = raw_value - loc.to(device)
        else:
            outputs = (raw_value - loc.to(device)) / scale.to(device)
    else:
        outputs = raw_value
    return outputs


def rescale_output(outputs: ALL_NET_OUTPUT,
                   loc: Optional[torch.Tensor],
                   scale: Optional[torch.Tensor],
                   device: torch.device = torch.device('cpu')) -> ALL_NET_OUTPUT:
    """
    rescale the network output to its raw scale

    Args:
        outputs (ALL_NET_OUTPUT):
            network head output
        loc (Optional[torch.Tensor]):
            scaling location value
        scale (Optional[torch.Tensor]):
            scaling scale value
        device (torch.device):
            which device the output is stored

    Return:
        ALL_NET_OUTPUT:
            rescaleed network output
    """
    if isinstance(outputs, list) or isinstance(outputs, tuple):
        return [rescale_output(output, loc, scale, device) for output in outputs]
    if loc is not None or scale is not None:
        if isinstance(outputs, torch.distributions.Distribution):
            transform = AffineTransform(loc=0.0 if loc is None else loc.to(device),
                                        scale=1.0 if scale is None else scale.to(device),
                                        )
            outputs = TransformedDistribution_(outputs, [transform])
        else:
            if loc is None:
                outputs = outputs * scale.to(device)  # type: ignore[union-attr]
            elif scale is None:
                outputs = outputs + loc.to(device)
            else:
                outputs = outputs * scale.to(device) + loc.to(device)
    return outputs