# Search for the model
set -ex
python train_and_eval.py +benchmark=PEMS/pems04/pems04_12 +model=mixed_concat_darts

#Evauate the model
python test_evaluated_model.py +benchmark=PEMS/pems04/pems04_12 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems04/pems04_24 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems04/pems04_48 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=PEMS/pems04/pems04_96 +model=mixed_concat_darts