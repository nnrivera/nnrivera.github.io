import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from plotnine import ggplot, aes, geom_line, labs, theme_minimal
from sklearn.datasets import fetch_openml

import os
# =========================
# Load dataset
# =========================

#os.chdir("ACÁ PONER DIRECTORIO DONDE ESTAN LAS BASE DE DATOS")
feature_names = []

with open("spambase.names", "r") as f:
    for line in f:
        line = line.strip()

        # feature lines contain ':'
        if ":" in line:
            name = line.split(":")[0]
            feature_names.append(name)

# add target column name
column_names = feature_names + ["spam"]
column_names = column_names[1:]

# load dataset
spam = pd.read_csv(
    "spambase.data",
    header=None,
    names=column_names
)

X = spam.drop(columns="spam").values
y = spam["spam"].values.astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================
# Helper
# =========================
def staged_errors(model, Xtr, ytr, Xte, yte):
    train_err, test_err = [], []
    for y_tr_pred, y_te_pred in zip(model.staged_predict(Xtr),
                                   model.staged_predict(Xte)):
        train_err.append(zero_one_loss(ytr, y_tr_pred))
        test_err.append(zero_one_loss(yte, y_te_pred))
    return train_err, test_err

# =========================
# Models
# =========================
n_estimators = 500

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=0.5,
    random_state=0
)

gb = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)

# RF needs manual growth
def rf_errors(Xtr, ytr, Xte, yte, max_estimators=200):
    rf = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        random_state=0
    )
    tr, te = [], []

    for k in range(1, max_estimators + 1):
        rf.set_params(n_estimators=k)
        rf.fit(Xtr, ytr)

        tr.append(zero_one_loss(ytr, rf.predict(Xtr)))
        te.append(zero_one_loss(yte, rf.predict(Xte)))

    return tr, te

# =========================
# Fit models
# =========================
ada.fit(Xtr, ytr)
gb.fit(Xtr, ytr)

ada_tr, ada_te = staged_errors(ada, Xtr, ytr, Xte, yte)
gb_tr, gb_te = staged_errors(gb, Xtr, ytr, Xte, yte)
rf_tr, rf_te = rf_errors(Xtr, ytr, Xte, yte, n_estimators)

# =========================
# Build DataFrame
# =========================
def build_df(errors, model_name):
    return pd.DataFrame({
        "n_estimators": np.arange(1, len(errors) + 1),
        "error": errors,
        "model": model_name
    })

df_plot = pd.concat([
    build_df(ada_te, "AdaBoost"),
    build_df(gb_te, "Gradient Boosting"),
    build_df(rf_te, "Random Forest"),
])

# =========================
# Plot
# =========================
p = (
    ggplot(df_plot, aes(x="n_estimators", y="error", color="model"))
    + geom_line()
    + labs(
        title="Spam dataset: Error vs # estimators",
        x="# estimators",
        y="Test error"
    )
    + theme_minimal()
)

print(p)
