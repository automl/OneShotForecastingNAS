# Search for the model
set -ex
python train_and_eval.py +benchmark=LTSF/traffic/traffic_96 +model=mixed_concat_darts

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/traffic/traffic_96 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/traffic/traffic_192 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/traffic/traffic_336 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/traffic/traffic_720 +model=mixed_concat_darts
