"""Carrega e processa documentos de conciliacao bancaria."""
import os
import re
import unicodedata
import csv
from typing import List, Tuple

from langchain_core.documents import Document
import pandas as pd


def _ler_csv(file_path: str) -> pd.DataFrame:
    """Lê CSVs brasileiros com diferentes codificações e separadores."""
    ultimo_erro = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError as erro:
            ultimo_erro = erro

    raise ultimo_erro


def _ler_extrato_csv(file_path: str) -> pd.DataFrame:
    """Lê CSVs exportados do relatório bancário com linhas quebradas."""
    with open(file_path, "r", encoding="utf-8-sig", newline="") as arquivo:
        linhas = list(csv.reader(arquivo))

    inicio = next(
        (indice for indice, linha in enumerate(linhas)
         if linha and linha[0].strip().lower() == "lançamento"),
        None,
    )
    if inicio is None:
        return _ler_csv(file_path)

    registros = []
    descricao_pendente = ""
    for linha in linhas[inicio + 1:]:
        linha = linha + [""] * (4 - len(linha))
        primeiro, segundo, debito, credito = [campo.strip() for campo in linha[:4]]

        if primeiro and re.fullmatch(r"\d{2}/\d{2}/\d{4}", primeiro):
            descricao = " ".join(parte for parte in (descricao_pendente, segundo) if parte)
            valor = credito or debito
            if valor:
                registros.append({
                    "data": primeiro,
                    "descricao": descricao,
                    "valor": valor,
                    "tipo": "CREDITO" if credito else "DEBITO",
                })
            descricao_pendente = ""
        elif segundo:
            descricao_pendente = " ".join(
                parte for parte in (descricao_pendente, segundo) if parte
            )

    return pd.DataFrame(registros)


def _sem_acentos(texto: str) -> str:
    """Remove acentos para permitir reconhecer cabeçalhos brasileiros."""
    return "".join(
        caractere for caractere in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caractere)
    )


def _normalizar_dataframe(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Normaliza colunas e tipos de dados do DataFrame."""
    df.columns = [_sem_acentos(str(c).strip().lower()) for c in df.columns]

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
        elif lc in ["debito", "debitos"]:
            col_map[col] = "debito"
        elif lc in ["credito", "creditos"]:
            col_map[col] = "credito"

    df = df.rename(columns=col_map)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

    def converter_valores(coluna: pd.Series) -> pd.Series:
        return pd.to_numeric(
            coluna.astype(str)
            .str.replace("R$", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("d", "", case=False, regex=False)
            .str.strip(),
            errors="coerce"
        ).fillna(0.0)

    if "valor" in df.columns:
        df["valor"] = converter_valores(df["valor"])
    elif "debito" in df.columns or "credito" in df.columns:
        debito = converter_valores(df.get("debito", pd.Series(0, index=df.index)))
        credito = converter_valores(df.get("credito", pd.Series(0, index=df.index)))
        descricoes = df.get("descricao", pd.Series("", index=df.index)).astype(str).str.upper()
        entradas = descricoes.str.contains("RECEB|RESGATE|DEP DINHEIRO", regex=True)
        df["valor"] = credito.where(credito.ne(0), debito)
        df["valor"] = df["valor"].where(entradas, -df["valor"])
        df["tipo"] = entradas.map({True: "CREDITO", False: "DEBITO"})

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


def csv_para_documentos(
    file_path: str, doc_type: str, dataframe: pd.DataFrame = None
) -> List[Document]:
    """Converte cada linha do CSV em um Document LangChain."""
    df = dataframe if dataframe is not None else _ler_csv(file_path)
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
        df = _ler_extrato_csv(file_path)
        df = _normalizar_dataframe(df, "extrato")
        docs = csv_para_documentos(file_path, "extrato_bancario", df)
        return df, docs


def load_livro_razao(file_path: str) -> Tuple[pd.DataFrame, List[Document]]:
    """Carrega livro razao e retorna DataFrame + Documentos."""
    df = _ler_csv(file_path)
    df = _normalizar_dataframe(df, "razao")
    docs = csv_para_documentos(file_path, "livro_razao")
    return df, docs
