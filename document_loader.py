"""Carrega e processa documentos de conciliacao bancaria."""
import os
from typing import List, Tuple

from langchain_core.documents import Document
import pandas as pd


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
        df["valor"] = (
            df["valor"]
            .astype(str)
            .str.replace("R$", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
        df["tipo"] = df["tipo"].replace({
            "C": "CREDITO", "CRÉDITO": "CREDITO", "ENTRADA": "CREDITO",
            "D": "DEBITO", "DÉBITO": "DEBITO", "SAIDA": "DEBITO", "SAÍDA": "DEBITO"
        })

    return df


def csv_para_documentos(file_path: str, doc_type: str) -> List[Document]:
    """Converte cada linha do CSV em um Document LangChain."""
    df = pd.read_csv(file_path)
    df = _normalizar_dataframe(df, doc_type)
    file_name = os.path.basename(file_path)
    documents = []

    for idx, row in df.iterrows():
        partes = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        page_content = " | ".join(partes)

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


def load_extrato_bancario(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega extrato bancario e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df = _normalizar_dataframe(df, "extrato")
    docs = csv_para_documentos(file_path, "extrato_bancario")
    return df, docs


def load_livro_razao(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega livro razao e retorna DataFrame + Documentos."""
    df = pd.read_csv(file_path)
    df = _normalizar_dataframe(df, "razao")
    docs = csv_para_documentos(file_path, "livro_razao")
    return df, docs
