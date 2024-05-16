# Search for the model
python train_and_eval.py +benchmark=LTSF/ettm1/ettm1_96 +model=concat_darts/mixed_concat_darts_medium

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/ettm1/ettm1_96 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/ettm1/ettm1_192 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/ettm1/ettm1_336 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/ettm1/ettm1_720 +model=concat_darts/mixed_concat_darts_medium
