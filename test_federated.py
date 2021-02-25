
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.algo import Coordinator, Client


def gold_standard(X, y):
    regr = LogisticRegression(solver='lbfgs', C=1e9, max_iter=10000, fit_intercept=True).fit(X, y)
    coordinator = Coordinator()
    X, y, beta = coordinator.init(X, y)
    counter = 0
    max_iter = 10000
    finished = False
    while counter < max_iter and not finished:
        counter = counter + 1
        data_to_send = []
        try:
            data_to_send.append(coordinator.compute_derivatives(X, y, beta))
        except FloatingPointError:
            data_to_send.append("early_stop")
        beta, finished = coordinator.aggregate_beta(data_to_send)

        if finished:
            coordinator.set_coefs(beta)

            continue


def test_federated(X, y):
    X1 = X[:100, :]
    y1 = y[:100]
    df1 = pd.DataFrame(X1)
    df1["label"] = pd.Series(y1)
    df1.to_csv("client1.csv")
    coordinator = Coordinator()
    X1, y1, beta = coordinator.init(X1, y1)

    X2 = X[100:, :]
    y2 = y[100:]
    df2 = pd.DataFrame(X2)
    df2["label"] = pd.Series(y2)
    df2.to_csv("client2.csv")
    client = Client()
    X2, y2, beta = client.init(X2, y2)
    counter = 0
    max_iter = 10000
    finished = False
    while counter < max_iter and not finished:
        counter = counter + 1
        data_to_send = []
        try:
            data_to_send.append(coordinator.compute_derivatives(X1, y1, beta))
        except FloatingPointError:
            data_to_send.append("early_stop")
        try:
            data_to_send.append(client.compute_derivatives(X2, y2, beta))
        except FloatingPointError:
            data_to_send.append("early_stop")
        beta, finished = coordinator.aggregate_beta(data_to_send)

        if finished:
            coordinator.set_coefs(beta)
            print(coordinator.coef_)
            print(coordinator.predict(X))
            print(coordinator.predict_proba(X))
            client.set_coefs(beta)
            print(client.predict(X))
            print(client.predict_proba(X))
            continue


brca = load_breast_cancer()
X = brca.data[:, :10]
X = StandardScaler().fit_transform(X)
y = brca.target
gold_standard(X, y)
test_federated(X, y)
regr = LogisticRegression(solver='lbfgs', C=1e9, max_iter=10000, fit_intercept=True).fit(X, y)

