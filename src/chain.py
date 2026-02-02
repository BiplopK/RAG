from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from src.retriever.retriever import Retrivers
from src.config import *


load_dotenv()

store={}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_rag_chain(vectorstore):
    retriever=Retrivers(vectorstore)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    system_prompt=(
        "You are a helpful assistant.\n"
         "Rules:\n"
         "1) Use ONLY the given context to answer factual questions.\n"
         "2) If not found in context, say: 'I don't have that information in my dataset.'\n"
         "3) Answer in **plain text only**. Do NOT use Markdown formatting, bold, italics, or bullet points.\n"
         "3) Be clear and structured.\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human",
         "Context:\n{context}\n\n"
         "Question:\n{question}\n\n"
         "Answer:")
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda inputs: "\n\n".join(doc.page_content for doc, _ in retriever.retrive(inputs["question"]))
        )
        | prompt
        | llm
    )

    rag_chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    final_chain = rag_chain_with_memory | StrOutputParser()

    return final_chain