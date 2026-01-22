from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import *
from src.data_loader import load_data
from src.process_text import preprocess_text

def document_embeddings(embedding_model):

    # chunk_texts = [chunk.page_content for chunk in chunk_texts]
    # sentence_model="all-miniLM-L6-v2"
    # model=SentenceTransformer(sentence_model)

    # embeddings=model.encode(chunk_texts,show_progress_bar=True,convert_to_numpy=True)
    # return embeddings, model
    embeddings=HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings
