# Optimizing Time Series Forecasting Architectures: A Hierarchical Neural Architecture Search Approach
This repository contains all the codes required to reproduce the result of the paper: 
Optimizing Time Series Forecasting Architectures: A Hierarchical Neural Architecture Search Approach


## Usage
The dataset can be downloaded from https://github.com/thuml/Time-Series-Library?tab=readme-ov-file#usage
And you need to update the path to store the data and model under `experiments/configs/base.yaml`
```
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

We provide an example script showing how to search for the optimal architectures and evaluate the optimal architectures
```
cd experiments 
bash scripts/LTSF/etth1.sh
```
We store the optimal architectures under `experiments/ModelWeights` 
Therefore, you could directly evaluate the optimal architectures with the model found by our optimizer:
```
cd experiments
python test_evaluated_model.py +benchmark=LTSF/etth1/etth1_96 +model=mixed_concat_darts seed=0
```
