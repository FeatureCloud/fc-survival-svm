# Survival Support-Vector-Machine FeatureCloud App

## Description
A Survival SVM FeatureCloud app, allowing a federated training of a Survival SVM using a pure regression objective.

## Input
- `train`: containing the local training data (columns: features; rows: samples)
- `test`: containing the local test data

Those files should be CSV (comma-seperated values) files, but you can control which separator is used by setting `sep`. 
Survival analysis needs a survival time and a censoring state. These should be given as a column in the training and 
test files. The name of the label can be controlled by setting `label_survival_time` and `label_event`. Also give the 
value indicating that the event occurred (e.g. data is not censored) by setting `event_truth_value`.

## Output
- `model`: containing the pickled trained SVM object
- `pred`: containing the predictions generated on the local test data
- `train`: containing the local training data (copy of input)
- `test`: containing the local test data (copy of input)

### Loading the trained model
The pickled model is compatible with the `FastSurvivalSVM` class in 
[scikit-survival](https://github.com/sebp/scikit-survival), which is itself compatible with scikit-learn.

```python
import pickle
from sksurv.svm.survival_svm import FastSurvivalSVM

with open('model.pickle', 'r') as f:
    model: FastSurvivalSVM = pickle.load(f)

    # Load data ...
    # Preprocess data ...

    predictions = model.predict(X_test)

    # ...
```

## Workflows
Please note that it is advised to encode categorical features of the input e.g. using the one-hot scheme.
The One-hot Encoder FeatureCloud App can be used in a workflow prior to this app.

Can be combined with the following apps:
- Pre: Cross Validation, Normalization, Feature Selection, One-hot Encoder
- Post: Survival Regression Evaluation

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```yml
fc_survival_svm:
  privacy:
    enable_smpc: True  # SMPC enhances privacy in a trade-off for a longer runtime, by only sending masked output to the aggregator.
    min_samples: 3  # opt out when a split of data contains less than min_samples; can not be set lower than 3
  input:
    train: "train_encoded.csv"
    test: "test_encoded.csv"
  output:
    model: "model.pickle"
    meta: "meta.yml"
    pred: "pred.csv"
    train: "train.csv"  # optional, default: fc_survival_svm.input.train; filename name for a copy of the train input
    test: "test.csv"  # optional, default: fc_survival_svm.input.train; filename name for a copy of the test input
  format:
    sep: ","  # separator used in csv files
    label_survival_time: "tte"  # label for the time to event column
    label_event: "event"  # label for the event column
    event_truth_value: True  # optional, default=True; value of an entry in the event column when an event occurred
  split:
    mode: directory  # directory if cross validation was used before, else file
    dir: "cv"  # cv if cross validation app was used before, else .
  svm:  # only set these at coordinator; will be overwritten otherwise
    alpha: 1  # regularization parameter
    fit_intercept: False  # whether to fit an intercept or not
    max_iterations: 1000  # maximum number of iterations
```

## Privacy
- Exchanges the model parameters of the SVM
- Uses SMPC to exchange data - No local parameters are visible, only aggregations
