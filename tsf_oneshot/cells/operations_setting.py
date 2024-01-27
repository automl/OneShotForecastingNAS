ops_setting = {
    'small_dataset': {
        'seq_model': {
            'transformer': {'nhead': 4},

        },
        'flat_model': {
            'nbeats_g': {'width': 64,
                         'num_fc_layers': 2,
                         'thetas_dim': 32
                         },
            'nbeats_t': {'width': 64,
                         'num_fc_layers': 2,
                         'thetas_dim': 4
                         },
            'nbeats_s': {'width': 64,
                         'num_fc_layers': 2,
                         'thetas_dim': 4
                         },
        },
        'general': {'dropout': 0.4}
    },
    'other_dataset': {
        'seq_model': {},
        'flat_model': {},
        'general': {}
    },

}

"""
{
'seq_model': {
    'transformer': {'nhead': 8},
    'tcn': {'kernel_size': 7},

},
'flat_model': {
    'nbeats_g': {'width': 256,
                 'num_fc_layers': 2,
                 'thetas_dim': 128
                 },
    'nbeats_t': {'width': 256,
                 'num_fc_layers': 2,
                 'thetas_dim': 4
                 },
    'nbeats_s': {'width': 256,
                 'num_fc_layers': 2,
                 'thetas_dim': 4
                 },
},
'general': {'dropout': 0.2}
},
"""