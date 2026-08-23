"""Interface web do Alura Agent — Conciliação Bancária."""
import os
import tempfile
import streamlit as st
import pandas as pd
from document_loader import load_extrato_bancario, load_livro_razao
from conciliador import encontrar_matches, gerar_resumo_conciliacao
from agent import build_vector_store, load_existing_vector_store, ask_question

st.set_page_config(
    page_title="Alura Agent — Conciliação Bancária",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Alura Agent — Conciliação Bancária")
st.markdown("**Agente de IA para conciliação entre extrato bancário e livro razão**")
st.markdown("---")

# Inicializa estado
if "resumo_conciliacao" not in st.session_state:
    st.session_state.resumo_conciliacao = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conciliados" not in st.session_state:
    st.session_state.conciliados = pd.DataFrame()
if "nao_conc_ext" not in st.session_state:
    st.session_state.nao_conc_ext = pd.DataFrame()
if "nao_conc_raz" not in st.session_state:
    st.session_state.nao_conc_raz = pd.DataFrame()

# Sidebar — Uploads
with st.sidebar:
    st.header("📁 Documentos")
    st.info("Faça upload do extrato bancário e do livro razão.")

    extrato_file = st.file_uploader(
        "📄 Extrato Bancário (CSV)",
        type=["csv"],
        key="extrato"
    )

    razao_file = st.file_uploader(
        "📒 Livro Razão / Lançamentos Contábeis (CSV)",
        type=["csv"],
        key="razao"
    )

    st.markdown("---")

    # Configurações de matching
    with st.expander("⚙️ Configurações de conciliação"):
        tol_dias = st.slider("Tolerância de dias", 0, 10, 3)
        tol_valor = st.number_input("Tolerância de valor (R$)", 0.0, 10.0, 0.01, step=0.01)
        min_sim = st.slider("Similaridade mínima (descrição)", 0.0, 1.0, 0.4)

    st.markdown("---")
    st.markdown("**Tecnologias:**")
    st.markdown("- Python + LangChain + Pandas")
    st.markdown("- Gemini (Google AI)")
    st.markdown("- ChromaDB + HuggingFace Embeddings")

# Área principal
if extrato_file is None or razao_file is None:
    st.info("👈 Faça upload dos dois arquivos na barra lateral para iniciar a conciliação.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Extrato Bancário (CSV)")
        st.markdown("""
        Colunas esperadas:
        - **data** — Data da movimentação
        - **descricao** — Histórico / descrição
        - **valor** — Valor do lançamento
        - **tipo** — Crédito (C) ou Débito (D)
        - *(opcional)* **saldo** — Saldo acumulado
        """)
    with col2:
        st.markdown("### 📒 Livro Razão (CSV)")
        st.markdown("""
        Colunas esperadas:
        - **data** — Data do lançamento contábil
        - **descricao** — Histórico / descrição
        - **valor** — Valor do lançamento
        - **tipo** — Crédito (C) ou Débito (D)
        - *(opcional)* **conta** — Código da conta contábil
        - *(opcional)* **documento** — Número do documento
        """)

    st.markdown("---")
    st.markdown("### 💡 Exemplos de perguntas após a conciliação")
    st.markdown("""
    - *"Qual o saldo final do extrato?"*
    - *"Quais lançamentos do extrato não foram encontrados no razão?"*
    - *"Há diferenças de valor entre extrato e razão?"*
    - *"Quais cheques ainda não foram compensados?"*
    - *"Qual o total de tarifas bancárias no período?"*
    - *"Liste os lançamentos conciliados com divergência de data."*
    """)

else:
    # Processa os arquivos
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_ext:
        tmp_ext.write(extrato_file.getvalue())
        tmp_ext_path = tmp_ext.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_raz:
        tmp_raz.write(razao_file.getvalue())
        tmp_raz_path = tmp_raz.name

    try:
        df_ext, docs_ext = load_extrato_bancario(tmp_ext_path)
        df_raz, docs_raz = load_livro_razao(tmp_raz_path)

        # Executa conciliação
        conc, nao_ext, nao_raz = encontrar_matches(
            df_ext, df_raz,
            tolerancia_dias=tol_dias,
            tolerancia_valor=tol_valor,
            min_similaridade=min_sim
        )

        st.session_state.conciliados = conc
        st.session_state.nao_conc_ext = nao_ext
        st.session_state.nao_conc_raz = nao_raz

        # Gera resumo
        resumo = gerar_resumo_conciliacao(conc, nao_ext, nao_raz)
        st.session_state.resumo_conciliacao = resumo

        # Indexa documentos para o RAG
        all_docs = docs_ext + docs_raz
        build_vector_store(all_docs)

        # Remove arquivos temporários
        os.unlink(tmp_ext_path)
        os.unlink(tmp_raz_path)

    except Exception as e:
        st.error(f"❌ Erro ao processar arquivos: {str(e)}")
        os.unlink(tmp_ext_path)
        os.unlink(tmp_raz_path)
        st.stop()

    # Tabs: Resumo, Conciliados, Não Conciliados, Chat
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumo", "✅ Conciliados", "⚠️ Não Conciliados", "💬 Perguntar ao Agente"
    ])

    with tab1:
        st.subheader("Resumo da Conciliação")

        col1, col2, col3, col4 = st.columns(4)
        total_ext = len(conc) + len(nao_ext)
        total_raz = len(conc) + len(nao_raz)
        taxa = (len(conc) / max(total_ext, 1)) * 100

        col1.metric("Extrato", f"{total_ext} lanç.")
        col2.metric("Razão", f"{total_raz} lanç.")
        col3.metric("Conciliados", f"{len(conc)}")
        col4.metric("Taxa", f"{taxa:.1f}%")

        st.markdown("---")
        st.text_area("Resumo gerado", resumo, height=300, label_visibility="collapsed")

    with tab2:
        st.subheader("Lançamentos Conciliados")
        if not conc.empty:
            st.dataframe(conc, use_container_width=True)
            st.download_button(
                "📥 Baixar conciliados (CSV)",
                conc.to_csv(index=False).encode("utf-8"),
                "conciliados.csv",
                "text/csv"
            )
        else:
            st.info("Nenhum lançamento conciliado encontrado.")

    with tab3:
        st.subheader("Lançamentos Não Conciliados")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**No extrato, sem correspondência no razão:**")
            if not nao_ext.empty:
                st.dataframe(nao_ext, use_container_width=True)
                st.download_button(
                    "📥 Baixar não conciliados extrato",
                    nao_ext.to_csv(index=False).encode("utf-8"),
                    "nao_conciliados_extrato.csv",
                    "text/csv",
                    key="dl_ext"
                )
            else:
                st.success("Todos os lançamentos do extrato foram conciliados!")

        with col_b:
            st.markdown("**No razão, sem correspondência no extrato:**")
            if not nao_raz.empty:
                st.dataframe(nao_raz, use_container_width=True)
                st.download_button(
                    "📥 Baixar não conciliados razão",
                    nao_raz.to_csv(index=False).encode("utf-8"),
                    "nao_conciliados_razao.csv",
                    "text/csv",
                    key="dl_raz"
                )
            else:
                st.success("Todos os lançamentos do razão foram conciliados!")

    with tab4:
        st.subheader("💬 Pergunte ao Agente Contábil")
        st.info("O agente analisa os documentos e o resumo da conciliação para responder.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Digite sua pergunta sobre a conciliação...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analisando documentos..."):
                    try:
                        result = ask_question(question, st.session_state.resumo_conciliacao)
                        answer = result["answer"]
                        sources = result.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("📎 Fontes consultadas"):
                                for i, src in enumerate(sources[:5], 1):
                                    tipo = src.get("doc_type", "Desconhecido")
                                    file = src.get("file_name", "Desconhecido")
                                    st.markdown(f"**{i}.** `{tipo}` — `{file}`")

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
