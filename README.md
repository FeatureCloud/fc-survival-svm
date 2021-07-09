# Survival Support Vector Regression FeatureCloud App

## Description
A Survival SVM FeatureCloud app, allowing a federated training of a Survival SVM using a pure regression objective.

## Input
- `train`: containing the local training data (columns: features; rows: samples)
- `test`: containing the local test data

## Output
- `model`: containing a dump of the trained SVM object
- `pred`: containing the predictions generated on the local test data
- `test`: containing the local test data

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
  input:
    train: "train_encoded.csv"
    test: "test_encoded.csv"
  output:
    model: "model.pkl"
    pred: "pred.csv"
    test: "test.csv"
  format:
    sep: ","
    label_survival_time: "tte"
    label_event: "event"
    event_truth_value: True  # optional, default=True; value of an entry in the event column when a event occurred
  split:
    mode: directory  # directory if cross validation was used before, else file
    dir: data  # data if cross validation app was used before, else .
  svm:
    alpha: 1  # regularization parameter
    fit_intercept: False  # whether to fit an intercept or not
    max_iterations: 1000  # maximum number of iterations
```
