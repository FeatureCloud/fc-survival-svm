# Logistic Regression FeatureCloud App

## Description
A Logistic Regression FeautureCloud app, allowing a federated computation of the logistic regression algorithm.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data

## Output
- pred.csv containing the predicted value 
- prob.csv containing the predicted probability
- train.csv containing the local training data
- test.csv containing the local test data

## Workflows
Can be combined with the following apps:
- Pre: Cross Validation, Normalization, Feature Selection
- Post: Classification Evaluation

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_logistic_regression
  input:
    train: "train.csv"
    test: "test.csv"
  output:
    pred: "pred.csv"
    test: "test.csv"
  format:
    sep: ","
    label: "Class
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
  algo:
    max_iterations: 10000 # Maximum number of iterations
```
