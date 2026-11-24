import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from models import load_dataset

st.set_page_config(page_title="Clustering", layout="wide")
st.title("Clustering — K-Means")
st.caption("Kelompokkan data tanpa label. Pilih dataset dan jumlah cluster di sidebar.")

PILIHAN = ["Iris", "Wine", "Wine Quality (UCI)", "Diabetes"]

with st.sidebar:
    dataset = st.selectbox("Dataset", PILIHAN)
    k = st.slider("Jumlah cluster (k)", 2, 10, 3)
    random_state = int(st.number_input("Random state", 0, 10_000, 42, step=1))
    standardisasi = st.checkbox("Standardisasi fitur", value=True)
    st.caption("K-Means memakai n_init=10 dan inisialisasi k-means++.")


@st.cache_data
def get_data(nama):
    return load_dataset(nama)


def skor_silhouette(Xs, labels, random_state):
    """Silhouette dengan sampel maksimal 2000 baris agar tetap cepat."""
    if len(set(labels)) < 2:
        return float("nan")
    n = min(2000, len(Xs))
    return silhouette_score(Xs, labels, sample_size=n, random_state=random_state)


@st.cache_data
def kurva_k(nama, standardisasi, random_state):
    X, _, _ = get_data(nama)
    X = X.select_dtypes(include="number").dropna()
    Xs = StandardScaler().fit_transform(X) if standardisasi else X.to_numpy()
    baris = []
    for kk in range(2, 11):
        km = KMeans(n_clusters=kk, n_init=10, random_state=random_state).fit(Xs)
        baris.append({
            "k": kk,
            "inertia": km.inertia_,
            "silhouette": skor_silhouette(Xs, km.labels_, random_state),
        })
    return pd.DataFrame(baris)


try:
    X, y, target_names = get_data(dataset)
except Exception as e:
    st.error(f"Gagal memuat dataset **{dataset}**: {e}")
    st.stop()

X = X.select_dtypes(include="number")
mask = X.notna().all(axis=1)
X = X[mask]
y = y.loc[mask] if y is not None else None

if len(X) < k:
    st.error(f"Jumlah baris ({len(X)}) lebih sedikit dari k ({k}).")
    st.stop()

Xs = StandardScaler().fit_transform(X) if standardisasi else X.to_numpy()
km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
labels = km.labels_
sil = skor_silhouette(Xs, labels, random_state)

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah data", f"{len(X):,}")
c2.metric("Silhouette", "-" if np.isnan(sil) else f"{sil:.4f}")
if target_names is not None and y is not None:
    c3.metric("ARI vs label asli", f"{adjusted_rand_score(y, labels):.4f}")
else:
    c3.metric("Inertia", f"{km.inertia_:,.2f}")

with st.spinner("Menghitung kurva elbow dan silhouette..."):
    df_kurva = kurva_k(dataset, standardisasi, random_state)

kiri, kanan = st.columns(2)
with kiri:
    fig = px.line(df_kurva, x="k", y="inertia", markers=True, title="Elbow — inertia")
    fig.add_vline(x=k, line_dash="dash", line_color="crimson")
    st.plotly_chart(fig, width="stretch")
with kanan:
    fig = px.line(df_kurva, x="k", y="silhouette", markers=True, title="Silhouette per k")
    fig.add_vline(x=k, line_dash="dash", line_color="crimson")
    st.plotly_chart(fig, width="stretch")

# ---------- Sebaran cluster dalam 2D ----------
pca = PCA(n_components=2, random_state=random_state)
koord = pca.fit_transform(Xs)
df_pca = pd.DataFrame({
    "PC1": koord[:, 0],
    "PC2": koord[:, 1],
    "Cluster": [f"Cluster {v}" for v in labels],
})
fig = px.scatter(
    df_pca, x="PC1", y="PC2", color="Cluster", opacity=0.7,
    title=f"Sebaran cluster (PCA 2D) — total variansi {pca.explained_variance_ratio_.sum():.1%}",
)
pusat = pca.transform(km.cluster_centers_)
fig.add_scatter(
    x=pusat[:, 0], y=pusat[:, 1], mode="markers", name="Pusat cluster",
    marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
)
st.plotly_chart(fig, width="stretch")

# ---------- Profil tiap cluster ----------
st.subheader("Profil cluster")
df_profil = X.copy()
df_profil["Cluster"] = labels
profil = df_profil.groupby("Cluster").mean(numeric_only=True).round(3)
profil.insert(0, "Jumlah", df_profil.groupby("Cluster").size())
st.dataframe(profil, width="stretch")
st.caption(
    "Nilai pada tabel adalah rata-rata fitur asli per cluster (bukan hasil standardisasi)."
)
