from functools import partial

from tsf_oneshot.cells.encoders.components import (
    GRUEncoderModule,
    LSTMEncoderModule,
    TransformerEncoderModule,
    TCNEncoderModule,
    IdentityEncoderModule,
    MLPMixEncoderModule
)
from tsf_oneshot.cells.encoders.flat_components import (
    IdentityFlatEncoderModule, MLPFlatModule, NBEATSModule
)

PRIMITIVES_Encoder = {
    "gru": GRUEncoderModule,
    "lstm": LSTMEncoderModule,
    "transformer": TransformerEncoderModule,
    "tcn": TCNEncoderModule,
    "mlp_mix": MLPMixEncoderModule,
    "skip_connection": IdentityEncoderModule,
}

PRIMITIVES_FLAT_ENCODER = {
    'mlp': MLPFlatModule,
    'nbeats_g': partial(NBEATSModule, stack_type='g'),
    'nbeats_t': partial(NBEATSModule, stack_type='t', thetas_dim=4),
    'nbeats_s': partial(NBEATSModule, stack_type='s', thetas_dim=4),
    'skip_connection': IdentityFlatEncoderModule,
}



