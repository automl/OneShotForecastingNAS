# Search for the model
python train_and_eval.py +benchmark=LTSF/electricity/electricity_96 +model=mixed_concat_darts

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/electricity/electricity_96 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/electricity/electricity_192 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/electricity/electricity_336 +model=mixed_concat_darts
python test_evaluated_model.py +benchmark=LTSF/electricity/electricity_720 +model=mixed_concat_darts
