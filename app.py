"""Interface web do Alura Agent — Conciliacao Bancaria."""
import os
import tempfile
import streamlit as st
import pandas as pd
from document_loader import load_extrato_bancario, load_livro_razao
from conciliador import encontrar_matches, gerar_resumo_conciliacao
from agent import build_vector_store, load_existing_vector_store, ask_question, LANGCHAIN_STATUS, LANGCHAIN_ERROR

st.set_page_config(
    page_title="Alura Agent — Conciliacao Bancaria",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Alura Agent — Conciliacao Bancaria")
st.markdown("**Agente de IA para conciliacao entre extrato bancario e livro razao**")

# Debug: mostra qual modo esta ativo
if not LANGCHAIN_STATUS:
    st.warning(f"⚠️ LangChain nao disponivel. Usando modo fallback. Erro: {LANGCHAIN_ERROR}")
else:
    st.success("✅ LangChain carregado com sucesso!")

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

# Sidebar
with st.sidebar:
    st.header("📁 Documentos")
    st.info("Faca upload do extrato bancario e do livro razao.")

    extrato_file = st.file_uploader("📄 Extrato Bancario (CSV)", type=["csv"], key="extrato")
    razao_file = st.file_uploader("📒 Livro Razao (CSV)", type=["csv"], key="razao")

    st.markdown("---")

    with st.expander("⚙️ Configuracoes de conciliacao"):
        tol_dias = st.slider("Tolerancia de dias", 0, 10, 3)
        tol_valor = st.number_input("Tolerancia de valor (R$)", 0.0, 10.0, 0.01, step=0.01)
        min_sim = st.slider("Similaridade minima", 0.0, 1.0, 0.4)

    st.markdown("---")
    st.markdown("**Tecnologias:**")
    st.markdown("- Python + Gemini API")
    st.markdown("- ChromaDB + SentenceTransformers")
    if LANGCHAIN_STATUS:
        st.markdown("- LangChain (ativo)")
    else:
        st.markdown("- Modo fallback (API direta)")

# Area principal
if extrato_file is None or razao_file is None:
    st.info("👈 Faca upload dos dois arquivos na barra lateral para iniciar.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Extrato Bancario")
        st.markdown("Colunas: data, descricao, valor, tipo, saldo")
    with col2:
        st.markdown("### 📒 Livro Razao")
        st.markdown("Colunas: data, descricao, valor, tipo, conta, documento")

    st.markdown("---")
    st.markdown("### 💡 Exemplos de perguntas")
    st.markdown("- Qual o saldo final do extrato?")
    st.markdown("- Quais lancamentos nao foram conciliados?")
    st.markdown("- Qual o total de tarifas bancarias?")

else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_ext:
        tmp_ext.write(extrato_file.getvalue())
        tmp_ext_path = tmp_ext.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_raz:
        tmp_raz.write(razao_file.getvalue())
        tmp_raz_path = tmp_raz.name

    try:
        df_ext, docs_ext = load_extrato_bancario(tmp_ext_path)
        df_raz, docs_raz = load_livro_razao(tmp_raz_path)

        conc, nao_ext, nao_raz = encontrar_matches(
            df_ext, df_raz,
            tolerancia_dias=tol_dias,
            tolerancia_valor=tol_valor,
            min_similaridade=min_sim
        )

        st.session_state.conciliados = conc
        st.session_state.nao_conc_ext = nao_ext
        st.session_state.nao_conc_raz = nao_raz

        resumo = gerar_resumo_conciliacao(conc, nao_ext, nao_raz)
        st.session_state.resumo_conciliacao = resumo

        # Converte Documentos para dicts (funciona em ambos os modos)
        all_docs = []
        for doc in docs_ext + docs_raz:
            all_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        build_vector_store(all_docs)

        os.unlink(tmp_ext_path)
        os.unlink(tmp_raz_path)

    except Exception as e:
        st.error(f"❌ Erro ao processar arquivos: {str(e)}")
        os.unlink(tmp_ext_path)
        os.unlink(tmp_raz_path)
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumo", "✅ Conciliados", "⚠️ Nao Conciliados", "💬 Perguntar ao Agente"
    ])

    with tab1:
        st.subheader("Resumo da Conciliacao")

        col1, col2, col3, col4 = st.columns(4)
        total_ext = len(conc) + len(nao_ext)
        total_raz = len(conc) + len(nao_raz)
        taxa = (len(conc) / max(total_ext, 1)) * 100

        col1.metric("Extrato", f"{total_ext} lanc.")
        col2.metric("Razao", f"{total_raz} lanc.")
        col3.metric("Conciliados", f"{len(conc)}")
        col4.metric("Taxa", f"{taxa:.1f}%")

        st.markdown("---")
        st.text_area("Resumo gerado", resumo, height=300, label_visibility="collapsed")

    with tab2:
        st.subheader("Lancamentos Conciliados")
        if not conc.empty:
            st.dataframe(conc, use_container_width=True)
            st.download_button("📥 Baixar conciliados", conc.to_csv(index=False).encode("utf-8"), "conciliados.csv", "text/csv")
        else:
            st.info("Nenhum lancamento conciliado.")

    with tab3:
        st.subheader("Lancamentos Nao Conciliados")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**No extrato, sem correspondencia no razao:**")
            if not nao_ext.empty:
                st.dataframe(nao_ext, use_container_width=True)
                st.download_button("📥 Baixar", nao_ext.to_csv(index=False).encode("utf-8"), "nao_conc_ext.csv", "text/csv", key="dl_ext")
            else:
                st.success("Todos conciliados!")
        with col_b:
            st.markdown("**No razao, sem correspondencia no extrato:**")
            if not nao_raz.empty:
                st.dataframe(nao_raz, use_container_width=True)
                st.download_button("📥 Baixar", nao_raz.to_csv(index=False).encode("utf-8"), "nao_conc_raz.csv", "text/csv", key="dl_raz")
            else:
                st.success("Todos conciliados!")

    with tab4:
        st.subheader("💬 Pergunte ao Agente Contabil")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Digite sua pergunta sobre a conciliacao...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
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
