import tomllib
from pymongo import MongoClient

uri = tomllib.load(open(".streamlit/secrets.toml", "rb"))["mongo"]["uri"]
client = MongoClient(uri)
print(client.admin.command("ping"))