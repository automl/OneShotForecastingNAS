from tsf_oneshot.cells.decoders.components import (
    GRUDecoderModule,
    LSTMDecoderModule,
    MLPDecoderModule,
    TCNDecoderModule,
    TransformerDecoderModule,
    IdentityDecoderModule
)

PRIMITIVES_Decoder = {
    "gru": GRUDecoderModule,
    "lstm": LSTMDecoderModule,
    "transformer": TransformerDecoderModule,
    "mlp": MLPDecoderModule,
    "tcn": TCNDecoderModule,
    "skip_connection": IdentityDecoderModule,
}
