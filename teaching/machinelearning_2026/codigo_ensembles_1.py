import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from plotnine import ggplot, aes, geom_line, labs, theme_minimal

# =========================
# Dataset
# =========================
def generate_chi2_dataset(n=5000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    threshold = 9.34
    y = (np.sum(X**2, axis=1) > threshold).astype(int)
    return X, y

X, y = generate_chi2_dataset()
df = pd.DataFrame(X)
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
# AdaBoost
# =========================
n_estimators=400
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=0.5,
    random_state=0
)

ada.fit(Xtr, ytr)
ada_tr, ada_te = staged_errors(ada, Xtr, ytr, Xte, yte)

# =========================
# Gradient Boosting
# =========================
gb = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=0.1,
    max_depth=1,
    random_state=0
)

gb.fit(Xtr, ytr)
gb_tr, gb_te = staged_errors(gb, Xtr, ytr, Xte, yte)

# =========================
# Random Forest (manual growth)
# =========================
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

rf_tr, rf_te = rf_errors(Xtr, ytr, Xte, yte, max_estimators=n_estimators)

# =========================
# Build DataFrame (tidy)
# =========================
def build_df(errors, model_name, dataset="test"):
    return pd.DataFrame({
        "n_estimators": np.arange(1, len(errors) + 1),
        "error": errors,
        "model": model_name,
        "dataset": dataset
    })

df = pd.concat([
    build_df(ada_te, "AdaBoost"),
    build_df(gb_te, "Gradient Boosting"),
    build_df(rf_te, "Random Forest"),
])

# =========================
# Plot with plotnine
# =========================
p = (
    ggplot(df, aes(x="n_estimators", y="error", color="model"))
    + geom_line()
    + labs(
        title="Error vs Number of Estimators",
        x="# estimators",
        y="Test error"
    )
    + theme_minimal()
)

print(p)