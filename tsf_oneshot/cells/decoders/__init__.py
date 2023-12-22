from tsf_oneshot.cells.decoders.components import (
    GRUDecoderModule,
    LSTMDecoderModule,
    MLPMixDecoderModule,
    TCNDecoderModule,
    TransformerDecoderModule,
    IdentityDecoderModule
)

PRIMITIVES_Decoder = {
    "gru": GRUDecoderModule,
    "lstm": LSTMDecoderModule,
    "transformer": TransformerDecoderModule,
    "tcn": TCNDecoderModule,
    'mlp_mix': MLPMixDecoderModule,
    "skip_connection": IdentityDecoderModule,
}
