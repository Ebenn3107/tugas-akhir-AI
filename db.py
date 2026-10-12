import streamlit as st
from pymongo import MongoClient


@st.cache_resource
def get_client():
    return MongoClient(st.secrets["mongo"]["uri"], serverSelectionTimeoutMS=5000)


def ping():
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False
