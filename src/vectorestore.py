from langchain_chroma import Chroma
import os
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import *
def build_vectorestore(embeddings,documents):
    if os.path.exists(VECTORESTORE_PATH) and os.listdir(VECTORESTORE_PATH):
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=VECTORESTORE_PATH
        )
        print("Loading data !!!!!!")
    else:
        print("Creating Vector store !!!!!!")
        vectorstore=Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTORESTORE_PATH
        )
        
    return vectorstore