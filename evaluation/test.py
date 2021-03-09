import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.roc import compute_roc_params

dir = "../test/brca_norm.csv"

data = pd.read_csv(dir)
target = data.loc[:, "label"]

data = data.drop("label", axis=1)
data = StandardScaler().fit_transform(data)

X, X_test, y, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
model = LogisticRegression(penalty="none", solver='lbfgs', C=1e9, max_iter=1000, fit_intercept=True).fit(X, y)
y_proba = model.predict_proba(X_test)[:, 1]

fpr_sklearn, tpr_sklearn, thr_sklearn = roc_curve(y_test, y_proba)
print(fpr_sklearn)

y_test1, y_proba1 = y_test[-100:], y_proba[-100:]
y_test2, y_proba2 = y_test[100:], y_proba[100:]
fpr_fc1, tpr_fc1, thr_fc1 = compute_roc_params(y_test1, y_proba1)
fpr_fc2, tpr_fc2, thr_fc2 = compute_roc_params(y_test2, y_proba2)

client1 = pd.DataFrame(data=[fpr_fc1, tpr_fc1, thr_fc1]).transpose()
client1.columns = ["FPR", "TPR", "THR"]
client1 = client1.set_index("THR")
client2 = pd.DataFrame(data=[fpr_fc2, tpr_fc2, thr_fc2]).transpose()
client2.columns = ["FPR", "TPR", "THR"]
client2 = client2.set_index("THR")
print(client1.index.tolist())
print(client2.index.tolist())
thr_global = sorted(set(client1.index.tolist() + client2.index.tolist()))
thr_global.reverse()
print(thr_global)

clients = [client1, client2]
roc = []
fprs = []
tprs = []
for thr in thr_global:
    fpr = []
    tpr = []
    for i in range(len(clients)):
        try:
            fpr.append(clients[i].loc[thr, "FPR"])
        except KeyError:
            fpr.append(fpr_previous[i])
        try:
            tpr.append(clients[i].loc[thr, "TPR"])
        except KeyError:
            tpr.append(tpr_previous[i])
    fprs.append(np.mean(fpr))
    tprs.append(np.mean(tpr))
    fpr_previous = fpr
    tpr_previous = tpr

global_roc = pd.DataFrame(index=thr_global)
global_roc["FPR"] = fprs
global_roc["TPR"] = tprs
print(global_roc)