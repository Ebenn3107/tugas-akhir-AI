from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Nilai = class; instance dibuat saat train supaya random_state bisa dipasang.
REGRESSION_MODELS = {
    "Linear": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier,
    "SVC": SVC,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
}

DATASETS = {
    "Regresi": ["Diabetes", "California Housing"],
    "Klasifikasi": ["Iris", "Wine"],
}

_LOADERS = {
    "Diabetes": datasets.load_diabetes,
    "California Housing": datasets.fetch_california_housing,  # download sekali, lalu di-cache sklearn
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
}


def load_dataset(nama):
    """Return (X: DataFrame, y: Series, target_names | None)."""
    data = _LOADERS[nama](as_frame=True)
    return data.data, data.target, getattr(data, "target_names", None)
