import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.algo import Coordinator, Client


def parse_input(path):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = pd.read_csv(path, sep=",").select_dtypes(include=numerics).dropna()
    y = X.loc[:, "label"]
    X = X.drop("label", axis=1)

    return X, y


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = parse_input("brca_norm.csv")
        self.global_model = LogisticRegression(penalty="none", solver='lbfgs', max_iter=10000,
                                               fit_intercept=True).fit(self.X, self.y)
        print(self.global_model.intercept_)

        X1, y1 = parse_input("client1/client.csv")
        self.coordinator = Coordinator()
        X1, y1, beta = self.coordinator.init(X1, y1)

        X2, y2 = parse_input("client2/client.csv")
        self.client = Client()
        X2, y2, beta = self.client.init(X2, y2)

        X3, y3 = parse_input("client3/client3.csv")
        self.client2 = Client()
        X3, y3, beta = self.client.init(X3, y3)

        counter = 0
        max_iter = 10000
        finished = False
        while counter < max_iter and not finished:
            counter = counter + 1
            data_to_send = []
            try:
                data_to_send.append(self.coordinator.compute_derivatives(X1, y1, beta))
            except FloatingPointError:
                data_to_send.append("early_stop")
            try:
                data_to_send.append(self.client.compute_derivatives(X2, y2, beta))
            except FloatingPointError:
                data_to_send.append("early_stop")
            try:
                data_to_send.append(self.client.compute_derivatives(X3, y3, beta))
            except FloatingPointError:
                data_to_send.append("early_stop")
            beta, finished = self.coordinator.aggregate_beta(data_to_send)

            if finished:
                self.coordinator.set_coefs(beta)
                print(self.coordinator.intercept_)
                self.client.set_coefs(beta)
                print(self.client.intercept_)
                continue

    def test_score(self):
        score_global = self.global_model.score(self.X, self.y)
        score1 = self.client.score(self.X, self.y)
        score2 = self.coordinator.score(self.X, self.y)
        print(score_global)
        print(score1)
        print(score2)
        np.testing.assert_allclose(score_global, score1)
        np.testing.assert_allclose(score2, score1)
        np.testing.assert_allclose(score2, score_global)


if __name__ == "__main__":
    unittest.main()
