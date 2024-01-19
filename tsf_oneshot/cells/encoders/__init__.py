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
    IdentityFlatEncoderModule, MLPFlatModule, NBEATSModule, NHitsModule, NBEATS_DEFAULT_THETA_DIMS
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
    'nbeats_g': partial(NBEATSModule, stack_type='g', thetas_dim=NBEATS_DEFAULT_THETA_DIMS['g']),
    'nbeats_t': partial(NBEATSModule, stack_type='t', thetas_dim=NBEATS_DEFAULT_THETA_DIMS['t']),
    'nbeats_s': partial(NBEATSModule, stack_type='s', thetas_dim=NBEATS_DEFAULT_THETA_DIMS['s']),
    'nhits_l': partial(NHitsModule, interpolation_mode='linear'),
    'nhits_n': partial(NHitsModule, interpolation_mode='nearest'),
    'nhits_c': partial(NHitsModule, interpolation_mode='cubic'),
    'skip_connection': IdentityFlatEncoderModule,
}



