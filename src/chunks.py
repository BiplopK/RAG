from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import *


def get_chunks(texts):
    document = Document(page_content=texts)
    documents=[document]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n","."],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunk_texts = text_splitter.split_documents(documents)
    return chunk_texts