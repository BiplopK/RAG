from langchain_huggingface import HuggingFaceEmbeddings

from src.config import *

def document_embeddings(embedding_model):
    embeddings=HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings
