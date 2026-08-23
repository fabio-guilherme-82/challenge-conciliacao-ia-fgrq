"""Carrega e processa documentos de conciliacao bancaria."""
import os
import re
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
            "C": "CREDITO", "CREDITO": "CREDITO", "ENTRADA": "CREDITO",
            "D": "DEBITO", "DEBITO": "DEBITO", "SAIDA": "DEBITO", "SAIDA": "DEBITO"
        })

    return df


def _extrair_tabela_pdf(texto: str) -> pd.DataFrame:
    """Tenta extrair uma tabela de dados de um texto de PDF de extrato bancario."""
    linhas = texto.split("\n")
    registros = []

    # Padroes comuns de linha de extrato: DATA | DESCRICAO | VALOR | SALDO
    padrao = re.compile(
        r"(\d{2}[/.-]\d{2}[/.-]\d{2,4})\s+(.+?)\s+([\d.,]+)\s+([\d.,]+)"
    )

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        match = padrao.search(linha)
        if match:
            data_str, descricao, valor_str, saldo_str = match.groups()
            # Tenta identificar se e debito ou credito pela descricao ou valor
            tipo = "CREDITO" if "CRED" in descricao.upper() or "RECEB" in descricao.upper() else "DEBITO"
            registros.append({
                "data": data_str,
                "descricao": descricao.strip(),
                "valor": valor_str.replace(".", "").replace(",", "."),
                "tipo": tipo,
                "saldo": saldo_str.replace(".", "").replace(",", ".")
            })

    if registros:
        return pd.DataFrame(registros)

    # Fallback: tenta encontrar qualquer linha com data e numero
    padrao_simples = re.compile(r"(\d{2}[/.-]\d{2}[/.-]\d{2,4})\s+(.+?)(\d+[.,]?\d*)")
    for linha in linhas:
        match = padrao_simples.search(linha)
        if match:
            data_str, descricao, valor_str = match.groups()
            registros.append({
                "data": data_str,
                "descricao": descricao.strip(),
                "valor": valor_str.replace(".", "").replace(",", "."),
                "tipo": "",
                "saldo": ""
            })

    return pd.DataFrame(registros) if registros else pd.DataFrame()


def load_pdf_extrato(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega um PDF de extrato bancario e retorna DataFrame + Documentos."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    texto_completo = ""
    for page in reader.pages:
        texto_completo += page.extract_text() or ""

    # Cria documentos para o RAG
    doc = Document(
        page_content=texto_completo,
        metadata={
            "source_type": "pdf",
            "doc_type": "extrato_bancario",
            "file_name": os.path.basename(file_path)
        }
    )
    docs = [doc]

    # Tenta extrair tabela estruturada
    df = _extrair_tabela_pdf(texto_completo)
    if df.empty:
        # Se nao conseguiu extrair tabela, cria um DataFrame vazio
        df = pd.DataFrame(columns=["data", "descricao", "valor", "tipo", "saldo"])
    else:
        df = _normalizar_dataframe(df, "extrato")

    return df, docs


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
    """Carrega extrato bancario (CSV ou PDF) e retorna DataFrame + Documentos."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf_extrato(file_path)
    else:
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
