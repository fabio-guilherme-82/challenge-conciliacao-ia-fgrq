"""Agente RAG para conciliacao bancaria usando Gemini direto + ChromaDB."""
import os
from typing import List, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
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

    vectorstore = Chroma.from_documents(
        documents=documents,
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
    """Faz uma pergunta ao agente RAG usando Gemini API direta."""
    import google.generativeai as genai

    vectorstore = load_existing_vector_store()
    if vectorstore is None:
        raise ValueError("Nenhum documento carregado. Faca upload dos arquivos primeiro.")

    api_key = _get_api_key()

    # 1. Busca documentos relevantes no ChromaDB
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs_relevantes = retriever.invoke(question)

    # 2. Monta o contexto
    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])

    # 3. Monta o prompt
    prompt = (
        "Voce e um assistente contabil especializado em conciliacao bancaria. "
        "Responda em portugues, com valores em R$. "
        "Se nao souber a resposta, diga que nao encontrou nos documentos. "
        "\n\nRESUMO DA CONCILIACAO:\n"
        + (resumo_conciliacao or "Nenhum resumo disponivel.") +
        "\n\nContexto dos documentos:\n"
        + contexto +
        "\n\nPergunta: " + question +
        "\n\nResposta:"
    )

    # 4. Chama a API do Gemini diretamente
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)
    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": [doc.metadata for doc in docs_relevantes]
    }
