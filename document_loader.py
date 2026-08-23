"""Carrega e processa documentos de conciliação bancária."""
import os
from typing import List, Tuple

# --- Tenta importar Document do LangChain ---
try:
    from langchain_core.documents import Document
    LANGCHAIN_DOC = True
except Exception:
    LANGCHAIN_DOC = False
    # Classe Document simples para fallback
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_PDF = True
except Exception:
    LANGCHAIN_PDF = False

import pandas as pd


def load_pdf(file_path: str) -> List[Document]:
    """Carrega um PDF e retorna lista de Document."""
    if LANGCHAIN_PDF:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["file_name"] = os.path.basename(file_path)
        return documents
    else:
        # Fallback simples: lê PDF como texto (requer pypdf)
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return [Document(
                page_content=text,
                metadata={"source_type": "pdf", "file_name": os.path.basename(file_path)}
            )]
        except Exception as e:
            raise ImportError(f"Não foi possível ler o PDF. Instale pypdf ou langchain-community. Erro: {e}")


def load_csv(file_path: str, doc_type: str = "generico") -> List[Document]:
    """Carrega um CSV e converte cada linha em um Document."""
    df = pd.read_csv(file_path)
    documents = []
    file_name = os.path.basename(file_path)

    for idx, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        page_content = " | ".join(parts)

        doc = Document(
            page_content=page_content,
            metadata={
                "source_type": "csv",
                "doc_type": doc_type,
                "file_name": file_name,
                "row_index": idx
            }
        )
        documents.append(doc)

    return documents


def load_document(file_path: str, doc_type: str = "generico") -> List[Document]:
    """Detecta o tipo de arquivo e carrega adequadamente."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".csv":
        return load_csv(file_path, doc_type=doc_type)
    else:
        raise ValueError(f"Formato não suportado: {ext}. Use PDF ou CSV.")


def _normalizar_dataframe(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Normaliza colunas e tipos de dados do DataFrame."""
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ["data", "date", "dt", "data_mov", "data_movimentacao", "data_transacao", "data_lancamento"]:
            col_map[col] = "data"
        elif lc in ["descricao", "historico", "desc", "detalhe", "historico_movimentacao", "descricao_lancamento", "historico_lancamento"]:
            col_map[col] = "descricao"
        elif lc in ["valor", "vlr", "amount", "val", "valor_movimentacao", "valor_lancamento"]:
            col_map[col] = "valor"
        elif lc in ["tipo", "natureza", "operacao", "debito_credito", "dc"]:
            col_map[col] = "tipo"
        elif lc in ["saldo", "saldo_atual", "balance"]:
            col_map[col] = "saldo"
        elif lc in ["conta", "conta_contabil", "codigo_conta"]:
            col_map[col] = "conta"
        elif lc in ["documento", "doc", "numero_documento", "ndoc"]:
            col_map[col] = "documento"

    df = df.rename(columns=col_map)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

    if "valor" in df.columns:
        df["valor"] = df["valor"].astype(str).str.replace("R$", "", regex=False).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)

    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
        df["tipo"] = df["tipo"].replace({
            "C": "CREDITO", "CRÉDITO": "CREDITO", "ENTRADA": "CREDITO",
            "D": "DEBITO", "DÉBITO": "DEBITO", "SAIDA": "DEBITO", "SAÍDA": "DEBITO"
        })

    return df


def load_extrato_bancario(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega extrato bancário e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df = _normalizar_dataframe(df, "extrato")
    docs = load_document(file_path, doc_type="extrato_bancario")
    return df, docs


def load_livro_razao(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega livro razão/lançamentos contábeis e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df = _normalizar_dataframe(df, "razao")
    docs = load_document(file_path, doc_type="livro_razao")
    return df, docs
