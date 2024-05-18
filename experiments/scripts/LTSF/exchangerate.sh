# Search for the model
set -ex
python train_and_eval.py +benchmark=LTSF/exchange_rate/exchange_rate_96 +model=concat_darts/mixed_concat_darts_medium

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/exchange_rate/exchange_rate_96 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/exchange_rate/exchange_rate_192 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/exchange_rate/exchange_rate_336 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/exchange_rate/exchange_rate_720 +model=concat_darts/mixed_concat_darts_small