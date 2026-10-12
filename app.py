import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import db
from models import CLASSIFICATION_MODELS, DATASETS, REGRESSION_MODELS, load_dataset

st.set_page_config(page_title="ML Playground", layout="wide")
st.title("ML Playground")

# ---------- Sidebar ----------
with st.sidebar:
    jenis = st.radio("Jenis", ["Regresi", "Klasifikasi"])
    dataset = st.selectbox("Dataset", DATASETS[jenis])
    model_dict = REGRESSION_MODELS if jenis == "Regresi" else CLASSIFICATION_MODELS
    model_name = st.selectbox("Model", list(model_dict))
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = int(st.number_input("Random state", 0, 10_000, 42, step=1))
    train = st.button("Train", type="primary", width="stretch")

    st.divider()
    if db.ping():
        st.success("MongoDB: terhubung")
    else:
        st.error("MongoDB: gagal terhubung")


@st.cache_data
def get_data(nama):
    return load_dataset(nama)


# ---------- Train ----------
if train:
    X, y, target_names = get_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if jenis == "Klasifikasi" else None,
    )
    est = model_dict[model_name]()
    if "random_state" in est.get_params():
        est.set_params(random_state=random_state)
    model = make_pipeline(StandardScaler(), est).fit(X_train, y_train)

    st.session_state["result"] = {
        "jenis": jenis, "dataset": dataset, "model_name": model_name,
        "model": model, "X_test": X_test, "y_test": y_test,
        "y_pred": model.predict(X_test), "target_names": target_names,
    }

res = st.session_state.get("result")
if res is None:
    st.info("Pilih pengaturan di sidebar lalu klik **Train**.")
    st.stop()

st.caption(f"Model terakhir: **{res['model_name']}** pada **{res['dataset']}** ({res['jenis']})")
y_test, y_pred = res["y_test"], res["y_pred"]
tab_vis, tab_metric = st.tabs(["Training & Visualisasi", "Evaluation Metrics"])

# ---------- Tab 1 ----------
with tab_vis:
    if res["jenis"] == "Regresi":
        df = pd.DataFrame({"Aktual": y_test.values, "Prediksi": y_pred})
        fig = px.scatter(df, x="Aktual", y="Prediksi", opacity=0.6, title="Aktual vs Prediksi")
        lo, hi = df.min().min(), df.max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, width="stretch")

        df["Residual"] = df["Aktual"] - df["Prediksi"]
        st.plotly_chart(px.histogram(df, x="Residual", nbins=40, title="Distribusi Residual"),
                        width="stretch")
    else:
        X_test = res["X_test"]
        names = res["target_names"]
        df = X_test.iloc[:, :2].copy()
        df["Prediksi"] = [str(names[i]) for i in y_pred]
        df["Benar"] = np.where(y_pred == y_test.values, "Benar", "Salah")
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color="Prediksi", symbol="Benar",
                         title="Prediksi pada data test (2 fitur pertama)")
        st.plotly_chart(fig, width="stretch")

# ---------- Tab 2 ----------
with tab_metric:
    if res["jenis"] == "Regresi":
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        c3.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
    else:
        names = [str(n) for n in res["target_names"]]
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        cm = confusion_matrix(y_test, y_pred, labels=range(len(names)))
        fig = px.imshow(cm, x=names, y=names, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        title="Confusion Matrix")
        st.plotly_chart(fig, width="stretch")
