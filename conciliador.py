"""Lógica de conciliação bancária: matching e identificação de divergências."""
import pandas as pd
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


def similaridade_texto(a: str, b: str) -> float:
    """Retorna similaridade entre duas strings (0 a 1)."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def encontrar_matches(
    extrato: pd.DataFrame,
    razao: pd.DataFrame,
    tolerancia_dias: int = 3,
    tolerancia_valor: float = 0.01,
    min_similaridade: float = 0.4
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tenta fazer matching entre extrato bancário e livro razão.

    Retorna:
        - conciliados: matches encontrados
        - nao_conciliados_extrato: lançamentos do extrato sem correspondência
        - nao_conciliados_razao: lançamentos do razão sem correspondência
    """
    if extrato.empty or razao.empty:
        return pd.DataFrame(), extrato.copy(), razao.copy()

    extrato = extrato.copy()
    razao = razao.copy()

    extrato["_matched"] = False
    razao["_matched"] = False

    matches = []

    for idx_ext, row_ext in extrato.iterrows():
        if row_ext.get("_matched"):
            continue

        data_ext = row_ext.get("data")
        valor_ext = row_ext.get("valor")
        desc_ext = str(row_ext.get("descricao", ""))
        tipo_ext = str(row_ext.get("tipo", "")).upper()

        if pd.isna(valor_ext):
            continue

        melhor_score = 0
        melhor_idx = None

        for idx_raz, row_raz in razao.iterrows():
            if row_raz.get("_matched"):
                continue

            data_raz = row_raz.get("data")
            valor_raz = row_raz.get("valor")
            desc_raz = str(row_raz.get("descricao", ""))
            tipo_raz = str(row_raz.get("tipo", "")).upper()

            if pd.isna(valor_raz):
                continue

            # Critérios de matching
            valor_ok = abs(valor_ext - valor_raz) <= tolerancia_valor

            data_ok = True
            if pd.notna(data_ext) and pd.notna(data_raz):
                diff_dias = abs((data_ext - data_raz).days)
                data_ok = diff_dias <= tolerancia_dias

            tipo_ok = tipo_ext == tipo_raz or tipo_ext == "" or tipo_raz == ""

            sim = similaridade_texto(desc_ext, desc_raz)
            desc_ok = sim >= min_similaridade

            score = 0
            if valor_ok: score += 3
            if data_ok: score += 2
            if desc_ok: score += 2
            if tipo_ok: score += 1

            if score >= 5 and score > melhor_score:
                melhor_score = score
                melhor_idx = idx_raz

        if melhor_idx is not None:
            row_raz = razao.loc[melhor_idx]
            matches.append({
                "data_extrato": data_ext,
                "data_razao": row_raz.get("data"),
                "descricao_extrato": desc_ext,
                "descricao_razao": row_raz.get("descricao"),
                "valor_extrato": valor_ext,
                "valor_razao": row_raz.get("valor"),
                "tipo_extrato": tipo_ext,
                "tipo_razao": row_raz.get("tipo"),
                "documento_razao": row_raz.get("documento", ""),
                "conta_razao": row_raz.get("conta", ""),
                "score_matching": melhor_score,
                "diferenca_dias": abs((data_ext - row_raz.get("data")).days) if pd.notna(data_ext) and pd.notna(row_raz.get("data")) else None,
                "diferenca_valor": abs(valor_ext - row_raz.get("valor")) if pd.notna(row_raz.get("valor")) else None,
            })
            extrato.at[idx_ext, "_matched"] = True
            razao.at[melhor_idx, "_matched"] = True

    conciliados = pd.DataFrame(matches)
    nao_conciliados_extrato = extrato[~extrato["_matched"]].drop(columns=["_matched"]).reset_index(drop=True)
    nao_conciliados_razao = razao[~razao["_matched"]].drop(columns=["_matched"]).reset_index(drop=True)

    return conciliados, nao_conciliados_extrato, nao_conciliados_razao


def identificar_divergencias(
    conciliados: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Analisa os matches e identifica divergências específicas."""
    divergencias = {}

    if conciliados.empty:
        return divergencias

    # 1. Diferenças de data
    mask_data = conciliados["diferenca_dias"] > 0
    if mask_data.any():
        divergencias["diferenca_data"] = conciliados[mask_data][[
            "data_extrato", "data_razao", "descricao_extrato", "diferenca_dias"
        ]].copy()

    # 2. Diferenças de valor
    mask_valor = conciliados["diferenca_valor"] > 0.01
    if mask_valor.any():
        divergencias["diferenca_valor"] = conciliados[mask_valor][[
            "descricao_extrato", "valor_extrato", "valor_razao", "diferenca_valor"
        ]].copy()

    return divergencias


def gerar_resumo_conciliacao(
    conciliados: pd.DataFrame,
    nao_conciliados_extrato: pd.DataFrame,
    nao_conciliados_razao: pd.DataFrame
) -> str:
    """Gera um resumo textual da conciliação para o agente de IA."""
    total_ext = len(conciliados) + len(nao_conciliados_extrato)
    total_raz = len(conciliados) + len(nao_conciliados_razao)

    resumo = f"""=== RESUMO DA CONCILIAÇÃO BANCÁRIA ===

Total de lançamentos no extrato bancário: {total_ext}
Total de lançamentos no livro razão: {total_raz}
Lançamentos conciliados: {len(conciliados)}
Lançamentos NÃO conciliados no extrato: {len(nao_conciliados_extrato)}
Lançamentos NÃO conciliados no razão: {len(nao_conciliados_razao)}
Taxa de conciliação: {(len(conciliados)/max(total_ext,1)*100):.1f}%

"""

    if not nao_conciliados_extrato.empty:
        resumo += "LANÇAMENTOS DO EXTRATO SEM CORRESPONDÊNCIA NO RAZÃO:\n"
        for _, row in nao_conciliados_extrato.head(10).iterrows():
            data = row.get("data", "")
            desc = row.get("descricao", "")
            valor = row.get("valor", 0)
            tipo = row.get("tipo", "")
            resumo += f"  - {data} | {desc} | R$ {valor:,.2f} | {tipo}\n"
        if len(nao_conciliados_extrato) > 10:
            resumo += f"  ... e mais {len(nao_conciliados_extrato) - 10} lançamentos.\n"
        resumo += "\n"

    if not nao_conciliados_razao.empty:
        resumo += "LANÇAMENTOS DO RAZÃO SEM CORRESPONDÊNCIA NO EXTRATO:\n"
        for _, row in nao_conciliados_razao.head(10).iterrows():
            data = row.get("data", "")
            desc = row.get("descricao", "")
            valor = row.get("valor", 0)
            tipo = row.get("tipo", "")
            doc = row.get("documento", "")
            resumo += f"  - {data} | {desc} | R$ {valor:,.2f} | {tipo} | Doc: {doc}\n"
        if len(nao_conciliados_razao) > 10:
            resumo += f"  ... e mais {len(nao_conciliados_razao) - 10} lançamentos.\n"
        resumo += "\n"

    # Análise de divergências
    divergencias = identificar_divergencias(conciliados)
    if divergencias:
        resumo += "DIVERGÊNCIAS IDENTIFICADAS NOS LANÇAMENTOS CONCILIADOS:\n"
        for tipo_div, df_div in divergencias.items():
            resumo += f"\n{tipo_div.upper().replace('_', ' ')}: {len(df_div)} ocorrências\n"
            for _, row in df_div.head(5).iterrows():
                if tipo_div == "diferenca_data":
                    resumo += f"  - {row.get('descricao_extrato', '')}: diferença de {row.get('diferenca_dias', 0)} dias\n"
                elif tipo_div == "diferenca_valor":
                    resumo += f"  - {row.get('descricao_extrato', '')}: extrato R$ {row.get('valor_extrato', 0):,.2f} vs razão R$ {row.get('valor_razao', 0):,.2f}\n"
    else:
        resumo += "Nenhuma divergência significativa identificada nos lançamentos conciliados.\n"

    resumo += "\n=== FIM DO RESUMO ==="
    return resumo
