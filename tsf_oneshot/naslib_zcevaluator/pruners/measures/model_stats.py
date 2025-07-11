import copy
from tsf_oneshot.cells.encoders.flat_components import NBEATSModule
from tsf_oneshot.cells.encoders.components import TCNEncoderModule, SepTCNEncoderModule
from tsf_oneshot.cells.decoders.components import TCNDecoderModule, SepTCNDecoderModule
from torchtnt.utils.flops import FlopTensorDispatchMode

def get_model_flops(model,
                    inputs_past, inputs_future, clone_model=True):
    if clone_model:
        model = copy.deepcopy(model)

    # We need to remove or copy all the duplicated moduels
    for cell in model.flat_net.cells:
        for edge in cell.edges:
            module = cell.edges[edge]
            if isinstance(module, NBEATSModule):
                module.backcast_head = copy.deepcopy(module.backcast_head)
                module.forecast_head = copy.deepcopy(module.forecast_head)

    model.head = copy.deepcopy(model.head)
    for cell in model.seq_net.encoder.cells:
        for edge in cell.edges:
            module = cell.edges[edge]
            if isinstance(module, (TCNEncoderModule, SepTCNEncoderModule)):
                module.conv1 = None

    if hasattr(model.seq_net.decoder, 'cells'):
        for cell in model.seq_net.decoder.cells:
            for edge in cell.edges:
                module = cell.edges[edge]
                if isinstance(module, (TCNDecoderModule, SepTCNDecoderModule)):
                    module.conv1 = None

    # First we need to
    with FlopTensorDispatchMode(model) as ftdm:
        # model stats is doing some hooks so do it last
        _ = model(inputs_past, inputs_future)
        flops_forward = copy.deepcopy(ftdm.flop_counts)
        ftdm.reset()
    n_flops = 0
    for k_modules in flops_forward.keys():
        for k_ops in flops_forward[k_modules].keys():
            n_flops += flops_forward[k_modules][k_ops]
    return n_flops
