"""Logica de conciliacao bancaria: matching e deteccao de divergencias."""
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple


def similaridade_texto(a: str, b: str) -> float:
    """Retorna similaridade entre 0 e 1."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def conciliar(
    df_extrato: pd.DataFrame,
    df_razao: pd.DataFrame,
    tolerancia_dias: int = 3,
    tolerancia_valor: float = 0.01,
    similaridade_minima: float = 0.4
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Executa a conciliacao entre extrato e razao.
    Retorna: (conciliados, nao_conc_extrato, nao_conc_razao, divergencias, resumo_texto)
    """
    conciliados = []
    usados_extrato = set()
    usados_razao = set()

    for i, ext in df_extrato.iterrows():
        for j, raz in df_razao.iterrows():
            if j in usados_razao:
                continue

            # 1. Valor deve ser igual (dentro da tolerancia)
            if abs(ext.get("valor", 0) - raz.get("valor", 0)) > tolerancia_valor:
                continue

            # 2. Tipo deve ser compativel
            ext_tipo = str(ext.get("tipo", "")).upper()
            raz_tipo = str(raz.get("tipo", "")).upper()
            if ext_tipo and raz_tipo and ext_tipo != raz_tipo:
                continue

            # 3. Data dentro da tolerancia
            data_ok = False
            diff_dias = None
            if pd.notna(ext.get("data")) and pd.notna(raz.get("data")):
                diff_dias = abs((ext["data"] - raz["data"]).days)
                if diff_dias <= tolerancia_dias:
                    data_ok = True
            else:
                data_ok = True

            if not data_ok:
                continue

            # 4. Similaridade textual da descricao
            sim = similaridade_texto(ext.get("descricao", ""), raz.get("descricao", ""))
            if sim < similaridade_minima:
                continue

            # Match encontrado!
            usados_extrato.add(i)
            usados_razao.add(j)

            conciliados.append({
                "ext_data": ext.get("data"),
                "ext_descricao": ext.get("descricao"),
                "ext_valor": ext.get("valor"),
                "ext_tipo": ext.get("tipo"),
                "raz_data": raz.get("data"),
                "raz_descricao": raz.get("descricao"),
                "raz_valor": raz.get("valor"),
                "raz_tipo": raz.get("tipo"),
                "raz_conta": raz.get("conta"),
                "raz_documento": raz.get("documento"),
                "diferenca_dias": diff_dias,
                "diferenca_valor": abs(ext.get("valor", 0) - raz.get("valor", 0)),
                "similaridade_descricao": round(sim, 2)
            })
            break

    df_conciliados = pd.DataFrame(conciliados)

    nao_conciliados_extrato = df_extrato[~df_extrato.index.isin(usados_extrato)].copy()
    nao_conciliados_razao = df_razao[~df_razao.index.isin(usados_razao)].copy()

    # Divergencias: conciliados com diferenca significativa
    if not df_conciliados.empty:
        divergencias = df_conciliados[
            (df_conciliados["diferenca_dias"] > 0) |
            (df_conciliados["diferenca_valor"] > tolerancia_valor)
        ].copy()
    else:
        divergencias = pd.DataFrame()

    # Resumo em texto
    total_ext = len(df_extrato)
    total_raz = len(df_razao)
    total_conc = len(df_conciliados)
    taxa = (total_conc / total_ext * 100) if total_ext > 0 else 0

    resumo = (
        "RESUMO DA CONCILIACAO BANCARIA\n"
        "================================\n"
        f"Total de lancamentos no extrato: {total_ext}\n"
        f"Total de lancamentos no razao:   {total_raz}\n"
        f"Lancamentos conciliados:         {total_conc}\n"
        f"Taxa de conciliacao:             {taxa:.1f}%\n"
        f"Nao conciliados no extrato:      {len(nao_conciliados_extrato)}\n"
        f"Nao conciliados no razao:        {len(nao_conciliados_razao)}\n"
        f"Divergencias detectadas:         {len(divergencias)}\n\n"
        f"CRITERIOS UTILIZADOS:\n"
        f"- Tolerancia de dias: {tolerancia_dias}\n"
        f"- Tolerancia de valor: R$ {tolerancia_valor:.2f}\n"
        f"- Similaridade minima: {similaridade_minima:.0%}\n"
    )

    return df_conciliados, nao_conciliados_extrato, nao_conciliados_razao, divergencias, resumo
