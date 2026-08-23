"""Agente RAG para conciliacao bancaria."""
import os
from typing import List, Optional, Dict, Any

# --- Tenta importar LangChain (modo moderno LCEL) ---
LANGCHAIN_OK = False
LC_ERROR = ""

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_OK = True
except Exception as e:
    LC_ERROR = str(e)
    print(f"[AVISO] LangChain nao disponivel: {e}")
    print("[INFO] Usando fallback com API Gemini direta.")

# --- Imports sempre necessarios ---
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


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
    raise ValueError("GOOGLE_API_KEY nao encontrada. Configure em st.secrets ou variavel de ambiente.")


def _docs_to_dicts(documents):
    """Converte lista mista (Document ou dict) para lista de dicts."""
    result = []
    for d in documents:
        if hasattr(d, "page_content"):
            result.append({"content": d.page_content, "metadata": d.metadata})
        elif isinstance(d, dict):
            result.append(d)
        else:
            result.append({"content": str(d), "metadata": {}})
    return result


# ============================================================
# MODO 1: LangChain LCEL (se disponivel)
# ============================================================
if LANGCHAIN_OK:
    def build_vector_store_langchain(documents):
        """Usa LangChain + Chroma."""
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)

        docs = _docs_to_dicts(documents)
        lc_docs = [LCDocument(page_content=d["content"], metadata=d["metadata"]) for d in docs]

        vector_store = Chroma.from_documents(
            documents=lc_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )
        return vector_store

    def load_existing_vector_store_langchain():
        if not os.path.exists(CHROMA_PATH):
            return None
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )

    def ask_question_langchain(question, resumo_conciliacao=""):
        vector_store = load_existing_vector_store_langchain()
        if vector_store is None:
            raise ValueError("Nenhum documento carregado. Faca upload dos arquivos primeiro.")

        api_key = _get_api_key()
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})

        prompt_template = (
            "Voce e um assistente contabil especializado em conciliacao bancaria.

"
            "RESUMO DA CONCILIACAO:
{resumo}

"
            "Contexto:
{context}

"
            "Pergunta: {question}
"
            "Responda em portugues, com valores em R$. Se nao souber, diga que nao encontrou.

"
            "Resposta:"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "resumo"]
        )

        # Chain LCEL moderna
        def format_docs(docs):
            return "

".join([d.page_content for d in docs])

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "resumo": lambda x: resumo_conciliacao
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)

        # Recupera fontes
        source_docs = retriever.invoke(question)
        sources = [doc.metadata for doc in source_docs]

        return {"answer": answer, "sources": sources}


# ============================================================
# MODO 2: Fallback com API Gemini direta
# ============================================================
def build_vector_store_fallback(documents):
    """Usa ChromaDB direto + SentenceTransformers."""
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name="docs")
    except Exception:
        pass
    collection = client.create_collection(name="docs")

    docs = _docs_to_dicts(documents)
    if not docs:
        return collection

    texts = [d["content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    embeddings = get_embed_model().encode(texts).tolist()
    ids = [str(i) for i in range(len(docs))]

    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection


def load_existing_vector_store_fallback():
    if not os.path.exists(CHROMA_PATH):
        return None
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name="docs")
    except Exception:
        return None


def ask_question_fallback(question, resumo_conciliacao=""):
    collection = load_existing_vector_store_fallback()
    if collection is None:
        raise ValueError("Nenhum documento carregado. Faca upload dos arquivos primeiro.")

    api_key = _get_api_key()
    genai.configure(api_key=api_key)

    q_embed = get_embed_model().encode([question]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=8)

    context_chunks = []
    sources = []
    if results and results.get("documents"):
        for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
            for doc, meta in zip(doc_list, meta_list):
                context_chunks.append(doc)
                sources.append(meta)

    context = "

".join(context_chunks)

    prompt = (
        "Voce e um assistente contabil especializado em conciliacao bancaria.

"
        "RESUMO DA CONCILIACAO REALIZADA:
"
        + (resumo_conciliacao or "Nenhum resumo disponivel.") + "

"
        "Contexto dos documentos:
" + context + "

"
        "Pergunta: " + question + "
"
        "Responda em portugues, com valores em R$. Se nao souber, diga que nao encontrou.

"
        "Resposta:"
    )

    llm = genai.GenerativeModel(LLM_MODEL)
    response = llm.generate_content(prompt)

    return {"answer": response.text, "sources": sources}


# ============================================================
# API UNIFICADA
# ============================================================
def build_vector_store(documents):
    if LANGCHAIN_OK:
        return build_vector_store_langchain(documents)
    else:
        return build_vector_store_fallback(documents)


def load_existing_vector_store():
    if LANGCHAIN_OK:
        return load_existing_vector_store_langchain()
    else:
        return load_existing_vector_store_fallback()


def ask_question(question, resumo_conciliacao=""):
    if LANGCHAIN_OK:
        return ask_question_langchain(question, resumo_conciliacao)
    else:
        return ask_question_fallback(question, resumo_conciliacao)


# Exporta status para debug
LANGCHAIN_STATUS = LANGCHAIN_OK
LANGCHAIN_ERROR = LC_ERROR
