from datetime import timezone
from zoneinfo import ZoneInfo

import streamlit as st

import db

st.set_page_config(page_title="Komentar", layout="wide")
st.title("Komentar Publik")
st.caption("Tulis masukan tentang aplikasi. Semua komentar disimpan di MongoDB.")

WIB = ZoneInfo("Asia/Jakarta")


def waktu_wib(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(WIB).strftime("%d %b %Y %H:%M WIB")


with st.form("form_komentar", clear_on_submit=True):
    nama = st.text_input("Nama", max_chars=60, placeholder="Anonim")
    rating = st.slider("Rating", 1, 5, 5)
    isi = st.text_area("Komentar", max_chars=1000, height=120,
                       placeholder="Tulis pengalaman atau saran kamu...")
    kirim = st.form_submit_button("Kirim komentar", type="primary")

if kirim:
    if len(isi.strip()) < 5:
        st.warning("Komentar minimal 5 karakter.")
    else:
        try:
            db.insert_comment(nama, isi, rating)
            st.session_state["flash_komentar"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menyimpan komentar: {e}")

if st.session_state.pop("flash_komentar", None):
    st.success("Komentar tersimpan. Terima kasih!")

try:
    total = db.count_comments()
    rata = db.average_rating()
    daftar = db.load_comments(limit=50)
except Exception as e:
    st.error(
        "MongoDB tidak terhubung, jadi komentar belum bisa ditampilkan. "
        "Periksa `uri` di `.streamlit/secrets.toml` dan di menu **Secrets** "
        f"Streamlit Cloud. Detail: {e}"
    )
    st.stop()

c1, c2 = st.columns(2)
c1.metric("Total komentar", f"{total:,}")
c2.metric("Rata-rata rating", "-" if rata is None else f"{rata:.1f} / 5")

st.divider()
st.subheader("Komentar terbaru")

if not daftar:
    st.info("Belum ada komentar. Jadi yang pertama menulis!")
else:
    for k in daftar:
        with st.container(border=True):
            kol1, kol2 = st.columns([3, 1])
            with kol1:
                st.markdown(f"**{k.get('nama', 'Anonim')}**")
            with kol2:
                st.caption(waktu_wib(k.get("created_at")))
            st.markdown("⭐" * int(k.get("rating", 0)))
            st.write(k.get("komentar", ""))
