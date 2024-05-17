# Search for the model
set -ex
python train_and_eval.py +benchmark=PEMS/pems07/pems07_12 +model=mixed_concat_darts

#Evauate the model
python test_evaluated_model.py +benchmark=PEMS/pems07/pems07_12 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems07/pems07_24 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems07/pems07_48 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems07/pems07_96 +model=mixed_concat_darts