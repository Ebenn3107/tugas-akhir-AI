"""Logika upload CSV dan retraining untuk halaman Database & Retrain.

Dipisah dari halaman Streamlit supaya bisa diuji langsung tanpa UI.
"""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db
from models import CLASSIFICATION_MODELS, REGRESSION_MODELS


def normalisasi(nama_kolom):
    return str(nama_kolom).strip().lower().replace(" ", "_").replace("-", "_")


def slug(nama):
    return "".join(c if c.isalnum() else "_" for c in nama.lower()).strip("_")


def _varian(nama):
    """Beberapa bentuk nama kolom yang mungkin dipakai di CSV.

    Contoh Iris: "sepal length (cm)" diterima sebagai
    "sepal_length_(cm)", "sepal_length_cm", "sepal_length", atau "sepallength".
    """
    n = normalisasi(nama)
    tanpa_kurung = re.sub(r"\([^)]*\)", "", n).strip("_")
    tanpa_tanda = n.replace("(", "").replace(")", "").strip("_")
    kandidat = {n, tanpa_kurung, tanpa_tanda}
    kandidat |= {v.replace("/", "_") for v in list(kandidat)}
    kandidat |= {v.replace("/", "") for v in list(kandidat)}
    kandidat |= {v.replace("_", "") for v in list(kandidat)}
    return kandidat


def samakan_kolom(df, fitur):
    """Samakan nama kolom CSV dengan nama fitur dataset (spasi/underscore)."""
    peta = {}
    for c in fitur:
        for v in _varian(c):
            peta.setdefault(v, c)
    alias_target = {"target", "label", "kelas", "class", "species", "quality", "y"}
    rename = {}
    for c in df.columns:
        n = normalisasi(c)
        if n in peta:
            rename[c] = peta[n]
        elif n in alias_target and "target" not in df.columns:
            rename[c] = "target"
    return df.rename(columns=rename)


def encode_target(series, target_names):
    """Ubah label teks (setosa, versicolor, ...) menjadi angka 0..n-1."""
    if target_names is None:
        return pd.to_numeric(series, errors="coerce")
    peta = {str(n).strip().lower(): i for i, n in enumerate(target_names)}

    def konversi(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in peta:
            return peta[s]
        for nama, i in peta.items():
            if nama in s:
                return i
        try:
            f = float(v)
        except (TypeError, ValueError):
            return np.nan
        return int(f) if f.is_integer() and 0 <= f < len(target_names) else np.nan

    return series.map(konversi)


def buat_model(nama, jenis, random_state):
    est = (CLASSIFICATION_MODELS if jenis == "Klasifikasi" else REGRESSION_MODELS)[nama]()
    if "random_state" in est.get_params():
        est.set_params(random_state=random_state)
    return make_pipeline(StandardScaler(), est)


def latih_ulang(dataset, X, y, model_name, jenis, fitur, target_names,
                test_size, random_state):
    """Latih baseline (sklearn) dan model gabungan (sklearn + MongoDB)."""
    stratify = y if jenis == "Klasifikasi" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    baseline = buat_model(model_name, jenis, random_state).fit(X_tr, y_tr)
    pred_base = baseline.predict(X_te)

    df_db = db.load_uploads(dataset)
    if df_db.empty:
        raise ValueError("Belum ada data upload untuk dataset ini.")
    kurang = [c for c in fitur if c not in df_db.columns]
    if kurang:
        raise ValueError("Kolom data MongoDB tidak lengkap: " + ", ".join(kurang))

    X_db = df_db[fitur].apply(pd.to_numeric, errors="coerce")
    y_db = pd.to_numeric(df_db["target"], errors="coerce")
    valid = X_db.notna().all(axis=1) & y_db.notna()
    X_db, y_db = X_db[valid], y_db[valid]
    if X_db.empty:
        raise ValueError("Tidak ada baris MongoDB yang valid.")
    if jenis == "Klasifikasi":
        y_db = y_db.round().astype(int)

    X_gab = pd.concat([X_tr, X_db], ignore_index=True)
    y_gab = pd.concat([y_tr, y_db], ignore_index=True)
    gabungan = buat_model(model_name, jenis, random_state).fit(X_gab, y_gab)
    pred_gab = gabungan.predict(X_te)

    return {
        "baseline": baseline,
        "gabungan": gabungan,
        "y_te": y_te,
        "pred_base": pred_base,
        "pred_gab": pred_gab,
        "n_sklearn": len(X_tr),
        "n_db": len(X_db),
        "fitur": fitur,
        "target_names": target_names,
    }
