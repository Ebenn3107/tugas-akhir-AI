"""Koneksi dan operasi MongoDB untuk ML Playground.

Dua koleksi dipakai:
    uploads  - baris data yang diunggah pengguna, dipisah per nama dataset
    comments - komentar publik dari halaman Komentar
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

DB_NAME_DEFAULT = "tas_ai"


@st.cache_resource
def get_client():
    """Client MongoDB dari secrets, di-cache agar tidak membuka koneksi baru."""
    return MongoClient(
        st.secrets["mongo"]["uri"],
        serverSelectionTimeoutMS=5000,
        tz_aware=True,
    )


def get_db_name():
    return st.secrets["mongo"].get("db_name", DB_NAME_DEFAULT)


def get_db():
    return get_client()[get_db_name()]


def ping():
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


# ---------- Data upload ----------
def insert_uploads(dataset, rows):
    """Simpan baris upload. rows: list dict berisi 'features' dan 'target'."""
    if not rows:
        return 0
    docs = [
        {
            "dataset": dataset,
            "features": r["features"],
            "target": r["target"],
            "source": "upload",
            "uploaded_at": datetime.now(timezone.utc),
        }
        for r in rows
    ]
    return len(get_db()["uploads"].insert_many(docs).inserted_ids)


def load_uploads(dataset):
    """Ambil data upload satu dataset sebagai DataFrame (fitur diratakan)."""
    docs = list(get_db()["uploads"].find({"dataset": dataset}).sort("uploaded_at", 1))
    if not docs:
        return pd.DataFrame()
    rows = []
    for d in docs:
        row = {"_id": str(d["_id"]), "uploaded_at": d.get("uploaded_at")}
        row.update(d.get("features", {}))
        row["target"] = d.get("target")
        rows.append(row)
    return pd.DataFrame(rows)


def delete_upload(id_str):
    return get_db()["uploads"].delete_one({"_id": ObjectId(id_str)}).deleted_count


def count_uploads(dataset=None):
    query = {"dataset": dataset} if dataset else {}
    return get_db()["uploads"].count_documents(query)


# ---------- Komentar publik ----------
def insert_comment(nama, komentar, rating):
    doc = {
        "nama": (nama or "Anonim").strip()[:60] or "Anonim",
        "komentar": komentar.strip()[:1000],
        "rating": int(rating),
        "created_at": datetime.now(timezone.utc),
    }
    return get_db()["comments"].insert_one(doc).inserted_id


def load_comments(limit=50):
    return list(get_db()["comments"].find().sort("created_at", -1).limit(limit))


def count_comments():
    return get_db()["comments"].count_documents({})


def average_rating():
    hasil = list(
        get_db()["comments"].aggregate(
            [{"$group": {"_id": None, "rata": {"$avg": "$rating"}}}]
        )
    )
    return hasil[0]["rata"] if hasil else None
