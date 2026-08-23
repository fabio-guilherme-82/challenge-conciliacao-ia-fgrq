"""Agente RAG para conciliacao bancaria usando Google Embeddings + ChromaDB."""
import os
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_api_key():
    """Obtem a chave API do ambiente."""
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key
    raise ValueError("GOOGLE_API_KEY nao encontrada. Configure em st.secrets ou variavel de ambiente.")


def build_vector_store(documents: List[Document]):
    """Constroi o vector store com Google Embeddings."""
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    api_key = _get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    # Faz chunking dos documentos para melhor recuperacao
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore


def load_existing_vector_store():
    """Carrega vector store existente."""
    if not os.path.exists(CHROMA_PATH):
        return None
    api_key = _get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def ask_question(question: str, resumo_conciliacao: str = "") -> Dict[str, Any]:
    """Faz uma pergunta ao agente RAG."""
    vectorstore = load_existing_vector_store()
    if vectorstore is None:
        raise ValueError("Nenhum documento carregado. Faca upload dos arquivos primeiro.")

    api_key = _get_api_key()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=2048
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Voce e um assistente contabil especializado em conciliacao bancaria. "
            "Responda em portugues, com valores em R$. "
            "Se nao souber a resposta, diga que nao encontrou nos documentos. "
            "\n\nRESUMO DA CONCILIACAO:\n{resumo}\n\n"
            "Contexto dos documentos:\n{context}"
        )),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    resposta = rag_chain.invoke({
        "input": question,
        "resumo": resumo_conciliacao
    })

    return {
        "answer": resposta.get("answer", ""),
        "sources": [doc.metadata for doc in resposta.get("context", [])]
    }
