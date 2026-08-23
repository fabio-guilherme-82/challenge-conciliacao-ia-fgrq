"""Carrega e processa documentos de conciliação bancária."""
import os
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd


def load_pdf(file_path: str) -> List[Document]:
    """Carrega um PDF e retorna lista de Document do LangChain."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["file_name"] = os.path.basename(file_path)
    return documents


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


def load_extrato_bancario(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega extrato bancário e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normaliza nomes de colunas comuns
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ["data", "date", "dt", "data_mov", "data_movimentacao", "data_transacao"]:
            col_map[col] = "data"
        elif lc in ["descricao", "historico", "desc", "detalhe", "historico_movimentacao", "descricao_lancamento"]:
            col_map[col] = "descricao"
        elif lc in ["valor", "vlr", "amount", "val", "valor_movimentacao", "valor_lancamento"]:
            col_map[col] = "valor"
        elif lc in ["tipo", "natureza", "operacao", "debito_credito", "dc"]:
            col_map[col] = "tipo"
        elif lc in ["saldo", "saldo_atual", "balance"]:
            col_map[col] = "saldo"

    df = df.rename(columns=col_map)

    # Converte data
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

    # Converte valor
    if "valor" in df.columns:
        df["valor"] = df["valor"].astype(str).str.replace("R$", "").str.replace(".", "").str.replace(",", ".").astype(float)

    # Converte tipo para padronização
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
        df["tipo"] = df["tipo"].replace({
            "C": "CREDITO", "CRÉDITO": "CREDITO", "ENTRADA": "CREDITO",
            "D": "DEBITO", "DÉBITO": "DEBITO", "SAIDA": "DEBITO", "SAÍDA": "DEBITO"
        })

    docs = load_document(file_path, doc_type="extrato_bancario")
    return df, docs


def load_livro_razao(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega livro razão/lançamentos contábeis e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ["data", "date", "dt", "data_lancamento", "data_mov"]:
            col_map[col] = "data"
        elif lc in ["descricao", "historico", "desc", "detalhe", "historico_lancamento"]:
            col_map[col] = "descricao"
        elif lc in ["valor", "vlr", "amount", "val", "valor_lancamento"]:
            col_map[col] = "valor"
        elif lc in ["conta", "conta_contabil", "codigo_conta", "natureza"]:
            col_map[col] = "conta"
        elif lc in ["tipo", "debito_credito", "dc", "natureza"]:
            col_map[col] = "tipo"
        elif lc in ["documento", "doc", "numero_documento", "ndoc"]:
            col_map[col] = "documento"

    df = df.rename(columns=col_map)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

    if "valor" in df.columns:
        df["valor"] = df["valor"].astype(str).str.replace("R$", "").str.replace(".", "").str.replace(",", ".").astype(float)

    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
        df["tipo"] = df["tipo"].replace({
            "C": "CREDITO", "CRÉDITO": "CREDITO", "ENTRADA": "CREDITO",
            "D": "DEBITO", "DÉBITO": "DEBITO", "SAIDA": "DEBITO", "SAÍDA": "DEBITO"
        })

    docs = load_document(file_path, doc_type="livro_razao")
    return df, docs
