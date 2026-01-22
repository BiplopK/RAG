
from src.data_loader import load_data
from src.process_text import preprocess_text
from src.config import *
from src.embedding import document_embeddings
from src.chunks import get_chunks
from src.retriever.retriever import Retrivers
from src.vectorestore import build_vectorestore


if __name__== "__main__":

    data=load_data(DATASET_PATH)
    clean_text=preprocess_text(data)
    chunk_texts=get_chunks(data)
    embeddings=document_embeddings(EMBEDDING_MODEL)
    vectorstore=build_vectorestore(embeddings,chunk_texts)
    retriever=Retrivers(vectorstore)
    print("Welcome!!!")
    while True:
        query = input("\nUser: ")
        
        if query.lower() in ["exit", "stop"]:
            print("Goodbye! Have a Nice Day.")
            break
        
        retrieved_docs=retriever.retrive(query)
        if retrieved_docs:
            for doc,score in retrieved_docs:
                print(f"Text: {doc.page_content.strip()}")
                print("---------------")
        else:
            print("No content found")


        

    
    

    

    