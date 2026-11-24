import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import db
from models import CLASSIFICATION_MODELS, DATASETS, REGRESSION_MODELS, load_dataset
from retrain import encode_target, latih_ulang, samakan_kolom, slug

st.set_page_config(page_title="Database & Retrain", layout="wide")
st.title("Database & Retrain")
st.caption("Upload CSV ke MongoDB, lalu latih ulang model dengan data gabungan.")

JENIS = {nama: jenis for jenis, daftar in DATASETS.items() for nama in daftar}
SEMUA = list(JENIS)


@st.cache_data
def get_data(nama):
    return load_dataset(nama)


# ---------- Sidebar ----------
with st.sidebar:
    dataset = st.selectbox("Dataset", SEMUA)
    jenis = JENIS[dataset]
    st.caption(f"Jenis: **{jenis}**")
    st.divider()
    st.caption("Data upload disimpan di koleksi `uploads`, dipisah per dataset.")

try:
    X, y, target_names = get_data(dataset)
except Exception as e:
    st.error(f"Gagal memuat dataset **{dataset}**: {e}")
    st.stop()

fitur = list(X.columns)
target_names_list = [str(n) for n in target_names] if target_names is not None else None

st.subheader(f"Dataset: {dataset}")
c1, c2 = st.columns(2)
c1.metric("Baris scikit-learn", f"{len(X):,}")
try:
    jumlah_db = db.count_uploads(dataset)
except Exception:
    jumlah_db = None
c2.metric("Baris di MongoDB", "—" if jumlah_db is None else f"{jumlah_db:,}")

if jumlah_db is None:
    st.error(
        "MongoDB tidak terhubung. Periksa `uri` di `.streamlit/secrets.toml` dan di "
        "menu **Secrets** Streamlit Cloud, lalu reboot aplikasi."
    )

flash_upload = st.session_state.pop("flash_upload", None)
if flash_upload:
    st.success(f"{flash_upload:,} baris berhasil disimpan ke MongoDB.")
flash_hapus = st.session_state.pop("flash_hapus", None)
if flash_hapus:
    st.success("Baris berhasil dihapus dari MongoDB.")

# ---------- 1. Upload CSV ----------
with st.expander("1. Upload CSV", expanded=True):
    contoh = X.head(2).copy()
    if target_names_list is not None:
        contoh["target"] = [target_names_list[int(v)] for v in y.head(2)]
    else:
        contoh["target"] = y.head(2).values
    st.download_button(
        "Unduh template CSV",
        contoh.to_csv(index=False).encode("utf-8"),
        file_name=f"template_{slug(dataset)}.csv",
        mime="text/csv",
    )
    st.caption("Kolom wajib: " + ", ".join(fitur) + ", target")

    berkas = st.file_uploader("Pilih file CSV", type="csv")
    if berkas is not None:
        try:
            df = pd.read_csv(berkas)
        except Exception as e:
            st.error(f"CSV tidak bisa dibaca: {e}")
            df = None
        if df is not None:
            df = samakan_kolom(df, fitur)
            kurang = [c for c in fitur + ["target"] if c not in df.columns]
            if kurang:
                st.error("Kolom belum lengkap: " + ", ".join(kurang))
            else:
                df = df[fitur + ["target"]].copy()
                for c in fitur:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["target"] = encode_target(df["target"], target_names_list)
                jumlah_awal = len(df)
                df = df.dropna(subset=fitur + ["target"])
                if len(df) > 5000:
                    st.warning(f"Data dipotong ke 5.000 baris (dari {len(df):,} baris valid).")
                    df = df.head(5000)
                st.write(f"{len(df):,} baris siap disimpan (dari {jumlah_awal:,} baris CSV).")
                st.dataframe(df.head(20), width="stretch", hide_index=True)
                if df.empty:
                    st.warning("Tidak ada baris valid setelah pembersihan.")
                elif st.button("Simpan ke MongoDB", type="primary", disabled=jumlah_db is None):
                    rows = [
                        {
                            "features": {c: float(r[c]) for c in fitur},
                            "target": float(r["target"]) if target_names_list is None else int(r["target"]),
                        }
                        for _, r in df.iterrows()
                    ]
                    try:
                        n = db.insert_uploads(dataset, rows)
                        st.session_state["flash_upload"] = n
                        st.session_state.pop("retrain", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal menyimpan ke MongoDB: {e}")

# ---------- 2. Data di MongoDB ----------
with st.expander("2. Data di MongoDB", expanded=False):
    try:
        df_db = db.load_uploads(dataset)
    except Exception as e:
        st.error(f"Gagal membaca data dari MongoDB: {e}")
        df_db = pd.DataFrame()
    if df_db.empty:
        st.info("Belum ada data upload untuk dataset ini.")
    else:
        st.dataframe(df_db.drop(columns=["_id"]), width="stretch", hide_index=True)
        st.caption(f"{len(df_db):,} baris. Pilih satu _id untuk dihapus:")
        pilih = st.selectbox("_id", df_db["_id"].tolist(), format_func=lambda x: x[-8:])
        if st.button("Hapus baris terpilih"):
            try:
                db.delete_upload(pilih)
                st.session_state["flash_hapus"] = True
                st.session_state.pop("retrain", None)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal menghapus: {e}")

# ---------- 3. Retrain ----------
st.divider()
st.subheader("3. Retrain dan bandingkan")
st.caption(
    "Baseline dilatih di data sklearn. Model gabungan dilatih di data sklearn + data "
    "MongoDB, lalu keduanya diuji pada test set yang sama."
)

model_dict = CLASSIFICATION_MODELS if jenis == "Klasifikasi" else REGRESSION_MODELS
kol1, kol2, kol3 = st.columns(3)
with kol1:
    model_name = st.selectbox("Model", list(model_dict))
with kol2:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with kol3:
    random_state = int(st.number_input("Random state", 0, 10_000, 42, step=1))

bisa_retrain = bool(jumlah_db)
if st.button("Retrain Sekarang", type="primary", disabled=not bisa_retrain):
    with st.spinner("Melatih baseline dan model gabungan..."):
        try:
            hasil = latih_ulang(
                dataset, X, y, model_name, jenis, fitur, target_names_list,
                test_size, random_state,
            )
            st.session_state["retrain"] = {"dataset": dataset, "model": model_name, **hasil}
        except Exception as e:
            st.error(f"Retrain gagal: {e}")

if not bisa_retrain:
    st.info("Upload minimal satu baris data ke MongoDB untuk mengaktifkan retrain.")

hasil = st.session_state.get("retrain")
if hasil and hasil["dataset"] == dataset:
    st.caption(f"Model: **{hasil['model']}** · data latih: "
               f"{hasil['n_sklearn']:,} baris sklearn + {hasil['n_db']:,} baris MongoDB")
    y_te = hasil["y_te"]

    if jenis == "Regresi":
        def metrik(pred):
            return {
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "R²": r2_score(y_te, pred),
            }

        mb, mg = metrik(hasil["pred_base"]), metrik(hasil["pred_gab"])
        kiri, kanan = st.columns(2)
        with kiri:
            st.markdown("**Baseline (sklearn saja)**")
            for nama, nilai in mb.items():
                st.metric(nama, f"{nilai:.4f}")
        with kanan:
            st.markdown("**Setelah + data MongoDB**")
            for nama, nilai in mg.items():
                warna = "normal" if nama == "R²" else "inverse"
                st.metric(nama, f"{nilai:.4f}", delta=f"{nilai - mb[nama]:+.4f}",
                          delta_color=warna)
    else:
        acc_b = accuracy_score(y_te, hasil["pred_base"])
        acc_g = accuracy_score(y_te, hasil["pred_gab"])
        f1_b = f1_score(y_te, hasil["pred_base"], average="macro")
        f1_g = f1_score(y_te, hasil["pred_gab"], average="macro")
        kiri, kanan = st.columns(2)
        with kiri:
            st.markdown("**Baseline (sklearn saja)**")
            st.metric("Accuracy", f"{acc_b:.4f}")
            st.metric("Macro F1", f"{f1_b:.4f}")
        with kanan:
            st.markdown("**Setelah + data MongoDB**")
            st.metric("Accuracy", f"{acc_g:.4f}", delta=f"{acc_g - acc_b:+.4f}")
            st.metric("Macro F1", f"{f1_g:.4f}", delta=f"{f1_g - f1_b:+.4f}")

        labels = list(range(len(hasil["target_names"]))) if hasil["target_names"] else None
        for judul, pred in [("Baseline", hasil["pred_base"]),
                            ("Gabungan", hasil["pred_gab"])]:
            rep = classification_report(
                y_te, pred, labels=labels, target_names=hasil["target_names"],
                output_dict=True, zero_division=0,
            )
            st.markdown(f"**Classification report — {judul}**")
            st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    est = hasil["gabungan"][-1]
    if hasattr(est, "feature_importances_"):
        fi = pd.DataFrame({
            "Fitur": hasil["fitur"],
            "Importance": est.feature_importances_,
        }).sort_values("Importance")
        st.markdown("**Feature importance — model gabungan**")
        st.plotly_chart(
            px.bar(fi, x="Importance", y="Fitur", orientation="h",
                   title="Kontribusi fitur (Random Forest)"),
            width="stretch",
        )
