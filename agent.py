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


def _get_api_key() -> str:
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

    raise ValueError(
        "GOOGLE_API_KEY não encontrada. Configure em:
"
        "  • Streamlit Cloud: Advanced Settings > Secrets
"
        "  • Local: arquivo .env
"
        "  • OCI: variável de ambiente"
    )


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


def build_vector_store(documents: List[Document]) -> Chroma:
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


def load_existing_vector_store() -> Optional[Chroma]:
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def create_qa_chain(vector_store: Chroma, resumo_conciliacao: str = ""):
    llm = get_llm()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    prompt_template = f"""Você é um assistente contábil especializado em conciliação bancária. Sua função é analisar documentos financeiros e responder perguntas com precisão.

RESUMO DA CONCILIAÇÃO REALIZADA:
{{resumo_conciliacao}}

Contexto dos documentos (extrato bancário e livro razão):
{{context}}

Pergunta do usuário: {{question}}

Instruções:
- Responda em português de forma clara, objetiva e profissional.
- Baseie sua resposta APENAS nos documentos fornecidos e no resumo da conciliação.
- Para perguntas sobre divergências, cite valores e descrições específicas.
- Para perguntas sobre saldo, some ou subtraia os valores conforme o tipo (crédito/débito).
- Se a informação não estiver nos documentos, diga: "Não encontrei essa informação nos documentos fornecidos."
- Use formatação de moeda brasileira (R$) para valores.
- Seja conciso mas completo. Evite respostas genéricas.

Resposta:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "resumo_conciliacao"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def ask_question(question: str, resumo_conciliacao: str = "") -> dict:
    vector_store = load_existing_vector_store()
    if vector_store is None:
        raise ValueError("Nenhum documento foi carregado ainda. Faça o upload primeiro.")

    qa_chain = create_qa_chain(vector_store, resumo_conciliacao)
    result = qa_chain.invoke({"query": question, "resumo_conciliacao": resumo_conciliacao})

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result.get("source_documents", [])]
    }
