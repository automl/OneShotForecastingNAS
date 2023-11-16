from tsf_oneshot.cells.encoders.components import GRUEncoderModule, LSTMEncoderModule, TransformerEncoderModule, \
    TCNEncoderModule, IdentityEncoderModule

PRIMITIVES_Encoder = {
    "gru": GRUEncoderModule,
    "lstm": LSTMEncoderModule,
    "transformer": TransformerEncoderModule,
    "tcn": TCNEncoderModule,
    "skip_connection": IdentityEncoderModule,
}

