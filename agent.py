"""Agente RAG para conciliacao bancaria usando Gemini direto + ChromaDB."""
import os
from typing import List, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
LLM_MODEL = "gemini-2.0-flash"
GENERATION_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_supported_llm_model() -> str:
    """Escolhe o modelo Gemini que estiver disponível na conta do usuário."""
    try:
        import google.generativeai as genai

        api_key = _get_api_key()
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = []

        for model in models:
            name = getattr(model, "name", "")
            if not name:
                continue
            supported_methods = getattr(model, "supported_generation_methods", [])
            if "generateContent" in supported_methods:
                available.append(name.split("/")[-1])

        for candidate in GENERATION_MODELS:
            if candidate in available:
                return candidate

        if available:
            return available[0]
    except Exception:
        pass

    return LLM_MODEL


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

    documentos_por_fonte = {}
    for document in documents:
        conteudo = document.page_content.strip()
        if not conteudo:
            continue
        chave = document.metadata.get("doc_type", "documentos")
        documentos_por_fonte.setdefault(chave, []).append(conteudo)

    documentos_indexados = [
        Document(
            page_content="\n\n".join(conteudos),
            metadata={"doc_type": fonte},
        )
        for fonte, conteudos in documentos_por_fonte.items()
    ]
    if not documentos_indexados:
        raise ValueError("Nenhum conteúdo válido para gerar embeddings.")

    api_key = _get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    vectorstore = Chroma.from_documents(
        documents=documentos_indexados,
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


def _extract_text(response):
    """Extrai texto da resposta do Gemini mesmo quando a estrutura muda."""
    text = getattr(response, "text", None)
    if text:
        return text

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", [])
                if parts:
                    texts = []
                    for part in parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                    if texts:
                        return "".join(texts)
    except Exception:
        pass

    return str(response)


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
    llm_model = _get_supported_llm_model()
    llm = genai.GenerativeModel(llm_model)
    response = llm.generate_content(prompt)

    return {
        "answer": _extract_text(response),
        "sources": [doc.metadata for doc in docs_relevantes]
    }
