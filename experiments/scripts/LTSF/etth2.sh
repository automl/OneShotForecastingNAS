# Search for the model
python train_and_eval.py +benchmark=LTSF/ettm2/ettm2_96 +model=concat_darts/mixed_concat_darts_medium

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/ettm2/ettm2_96 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/ettm2/ettm2_192 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/ettm2/ettm2_336 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/ettm2/ettm2_720 +model=concat_darts/mixed_concat_darts_small
