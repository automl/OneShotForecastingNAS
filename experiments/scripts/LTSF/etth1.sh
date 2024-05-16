# Search for the model
python train_and_eval.py +benchmark=LTSF/etth1/etth1_96 +model=concat_darts/mixed_concat_darts_small

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/etth1/etth1_96 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/etth1/etth1_192 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/etth1/etth1_336 +model=concat_darts/mixed_concat_darts_small
python test_evaluated_model.py +benchmark=LTSF/etth1/etth1_720 +model=concat_darts/mixed_concat_darts_small
