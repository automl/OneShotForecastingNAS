# Search for the model
python train_and_eval.py +benchmark=LTSF/weather/weather_96 +model=concat_darts/mixed_concat_darts_medium

#Evauate the model
python test_evaluated_model.py +benchmark=LTSF/weather/weather_96 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/weather/weather_192 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/weather/weather_336 +model=concat_darts/mixed_concat_darts_medium
python test_evaluated_model.py +benchmark=LTSF/weather/weather_720 +model=concat_darts/mixed_concat_darts_medium
