import pandas as pd
import json
import re 
from langchain_core.documents import Document

from src.config import *

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for d in data:
        content = d.get("content", "")
        if isinstance(content, list):
            content = " ".join(content)
    return content