from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.process_text import preprocess_text


def get_chunks(texts):
    document = Document(page_content=texts)
    documents=[document]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n","."],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunk_texts = text_splitter.split_documents(documents)
    return chunk_texts