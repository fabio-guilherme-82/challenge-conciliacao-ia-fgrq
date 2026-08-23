"""Core do agente RAG para conciliação bancária."""
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"


def _get_api_key():
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

    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    raise ValueError("GOOGLE_API_KEY nao encontrada.")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    api_key = _get_api_key()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=2048,
    )


def build_vector_store(documents):
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def load_existing_vector_store():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def create_qa_chain(vector_store, resumo_conciliacao=""):
    llm = get_llm()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    prompt_template = (
        "Voce e um assistente contabil especializado em conciliacao bancaria. "
        "Sua funcao e analisar documentos financeiros e responder perguntas com precisao.\n\n"
        "RESUMO DA CONCILIACAO REALIZADA:\n"
        + resumo_conciliacao + "\n\n"
        "Contexto dos documentos (extrato bancario e livro razao):\n"
        "{context}\n\n"
        "Pergunta do usuario: {question}\n\n"
        "Instrucoes:\n"
        "- Responda em portugues de forma clara, objetiva e profissional.\n"
        "- Baseie sua resposta APENAS nos documentos fornecidos e no resumo da conciliacao.\n"
        "- Para perguntas sobre divergencias, cite valores e descricoes especificas.\n"
        "- Para perguntas sobre saldo, some ou subtraia os valores conforme o tipo (credito/debito).\n"
        "- Se a informacao nao estiver nos documentos, diga: 'Nao encontrei essa informacao nos documentos fornecidos.'\n"
        "- Use formatacao de moeda brasileira (R$) para valores.\n"
        "- Seja conciso mas completo. Evite respostas genericas.\n\n"
        "Resposta:"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def ask_question(question, resumo_conciliacao=""):
    vector_store = load_existing_vector_store()
    if vector_store is None:
        raise ValueError("Nenhum documento foi carregado ainda. Faca o upload primeiro.")

    qa_chain = create_qa_chain(vector_store, resumo_conciliacao)
    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result.get("source_documents", [])]
    }
