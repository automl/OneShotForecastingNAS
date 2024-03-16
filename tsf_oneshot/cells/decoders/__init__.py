from tsf_oneshot.cells.decoders.components import (
    GRUDecoderModule,
    LSTMDecoderModule,
    MLPMixDecoderModule,
    TCNDecoderModule,
    SepTCNDecoderModule,
    TransformerDecoderModule,
    IdentityDecoderModule
)

PRIMITIVES_Decoder = {
    "gru": GRUDecoderModule,
    "lstm": LSTMDecoderModule,
    "transformer": TransformerDecoderModule,
    "tcn": TCNDecoderModule,
    "sep_tcn": SepTCNDecoderModule,
    'mlp_mix': MLPMixDecoderModule,
    "skip_connection": IdentityDecoderModule,
}
