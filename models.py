from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ucimlrepo import fetch_ucirepo

# Nilai = class; instance dibuat saat train supaya random_state bisa dipasang.
REGRESSION_MODELS = {
    "Linear": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "MLP": lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=1200, early_stopping=True, random_state=42
    ),
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier,
    "SVC": SVC,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "MLP": lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1200, early_stopping=True, random_state=42
    ),
}

# Dataset UCI diambil lewat ucimlrepo; id mengikuti katalog UCI.
UCI_DATASETS = {
    "Wine Quality (UCI)": 186,
}

DATASETS = {
    "Regresi": ["Diabetes", "California Housing", "Wine Quality (UCI)"],
    "Klasifikasi": ["Iris", "Wine"],
}

JENIS_DATASET = {nama: jenis for jenis, daftar in DATASETS.items() for nama in daftar}

_LOADERS = {
    "Diabetes": datasets.load_diabetes,
    "California Housing": datasets.fetch_california_housing,  # download sekali, lalu di-cache sklearn
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
}


def _load_uci(nama):
    """Ambil dataset UCI dan kembalikan format yang sama dengan loader sklearn."""
    ds = fetch_ucirepo(id=UCI_DATASETS[nama])
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    return X, y, None


def load_dataset(nama):
    """Return (X: DataFrame, y: Series, target_names | None).

    target_names hanya diisi untuk dataset klasifikasi. Dataset regresi
    (Diabetes, California Housing) juga punya atribut target_names di sklearn,
    tetapi isinya nama target, bukan nama kelas.
    """
    if nama in UCI_DATASETS:
        return _load_uci(nama)
    data = _LOADERS[nama](as_frame=True)
    target_names = getattr(data, "target_names", None)
    if JENIS_DATASET.get(nama) != "Klasifikasi":
        target_names = None
    return data.data, data.target, target_names
