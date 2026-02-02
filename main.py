
from src.data_loader import load_data
from src.process_text import preprocess_text
from src.config import *
from src.embedding import document_embeddings
from src.chunks import get_chunks
from src.vectorestore import build_vectorestore
from src.generate_id import generate_session_id
from src.chain import build_rag_chain

if __name__== "__main__":
    print("WELCOME!!!!!")

    session_id = generate_session_id()
    print(f"Your generated Session ID: {session_id}\n")
    data=load_data(DATASET_PATH)
    clean_text=preprocess_text(data)
    chunk_texts=get_chunks(data)
    embeddings=document_embeddings(EMBEDDING_MODEL)
    vectorstore=build_vectorestore(embeddings,chunk_texts)

    chain = build_rag_chain(vectorstore)
    while True:
        user_query = input("User: ").strip()

        if user_query.lower() in ["stop", "exit", "quit"]:
            print("\nAssistant: Goodbye Have a nice day!")
            break

        answer = chain.invoke(
            {"question": user_query},
            config={"configurable": {"session_id": session_id}}
        )

        print("\nAssistant:", answer)
        print("\n")


        

    
    

    

    