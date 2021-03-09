import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eval import compute_min_max_score, agg_compute_thresholds, compute_threshold_conf_matrices, compute_roc_parameters, \
    compute_roc_auc, check

fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
tprs = []
fprs = []
thrs = []
aucs = []

dir = "/home/spaethju/Datasets/Workflows/Classification/tcga/central/tcga_all.csv"

data = pd.read_csv(dir)
target = data.loc[:, "mol_subt"]

data = data.drop("mol_subt", axis=1)
data = StandardScaler().fit_transform(data)

X, X_test, y, y_test = train_test_split(data, target, test_size=0.3)

for i in range(1):
    X, X_test, y, y_test = train_test_split(data, target, test_size=0.3)
    model = LogisticRegression(penalty="none", solver='lbfgs', C=1e9, max_iter=1000, fit_intercept=True).fit(X, y)
    y_proba = model.predict_proba(X_test)

    y_test, y_proba = check(y_test, y_proba)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    print(fpr.shape)
    print(tpr.shape)
    auc = roc_auc_score(y_test, y_proba)
    # tprs.append(tpr)
    # fprs.append(fpr)
    # aucs.append(auc)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Sklearn (AUC = {round(auc, 3)})', mode='lines'))

    min, max = compute_min_max_score(y_proba)
    thresholds_central = agg_compute_thresholds([[min, max]])
    confusion_matrices_central = compute_threshold_conf_matrices(y_test, y_proba, thresholds_central)
    roc_params_central = compute_roc_parameters(confusion_matrices_central, thresholds_central)
    tpr = np.array(roc_params_central["TPR"])
    fpr = np.array(roc_params_central["FPR"])
    thresholds = roc_params_central["THR"]
    print(fpr.shape)
    print(tpr.shape)
    if len(fpr) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fpr, 2),
                                                    np.diff(fpr, 2)),
                                      True])[0]
        print(optimal_idxs)
        fpr = fpr[optimal_idxs]
        tpr = tpr[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fpr[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fpr.shape)
    else:
        fpr = fpr / fpr[-1]

    if tpr[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tpr.shape)
    else:
        tpr = tpr / tpr[-1]
    print(fpr)
    print(tpr)
    auc = compute_roc_auc(fpr, tpr)
    df = pd.DataFrame(data=[fpr, tpr, thresholds]).transpose()
    df.columns = ["fpr", "tpr", "thresholds"]
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'FeatureCloud (AUC = {round(auc, 3)})', mode='lines'))

# mean_auc = round(np.mean(aucs, axis=0), 3)
# std_auc = round(np.std(aucs), 3)
# print(tprs)
# mean_tprs = np.mean(tprs, axis=0)
# mean_fprs = np.mean(fprs, axis=0)
#
# fig.add_trace(go.Scatter(x=mean_fprs, y=mean_tprs, name=f'Mean ROC (AUC = {mean_auc}) +/- {std_auc}', mode='lines'))

fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=700
)
fig.show()
