"""Conciliações Bank F.G.R.Q. - Conciliacao Bancaria com IA."""
import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import load_extrato_bancario, load_livro_razao
from conciliador import conciliar
from agent import build_vector_store, ask_question

# Configuracao da pagina
st.set_page_config(
    page_title="Conciliações Bank F.G.R.Q. | Conciliação Bancária",
    page_icon="🏦",
    layout="wide"
)

# Carrega variaveis de ambiente
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Estado da sessao
if "docs" not in st.session_state:
    st.session_state.docs = []
if "resumo" not in st.session_state:
    st.session_state.resumo = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("Configuracoes")
    st.markdown("---")

    tolerancia_dias = st.slider("Tolerancia de dias", 0, 10, 3)
    tolerancia_valor = st.slider("Tolerancia de valor (R$)", 0.0, 10.0, 0.01, 0.01)
    similaridade = st.slider("Similaridade minima", 0.0, 1.0, 0.4, 0.05)

    st.markdown("---")
    st.info("Desenvolvido para o Challenge Alura-Oracle ONE G10")
    st.markdown("**Autor:** Fabio Guilherme")

# Titulo principal
st.title("🏦 Conciliações Bank F.G.R.Q. - Conciliação Bancária")
st.markdown("Agente de IA especializado em conciliar extratos bancarios com o livro razao.")
st.markdown("**Agora com suporte a PDF!**")
st.markdown("---")

# Verifica API Key
if not api_key:
    st.error("GOOGLE_API_KEY nao configurada!")
    st.markdown("""
    Configure a chave de uma das seguintes formas:
    1. Crie um arquivo `.env` na raiz do projeto com: GOOGLE_API_KEY=sua_chave_aqui
    2. Ou configure no Streamlit Cloud em Settings -> Secrets
    """)
    st.stop()

# Abas
tab1, tab2, tab3 = st.tabs(["📤 Upload e Conciliacao", "📊 Resultados", "💬 Perguntar ao Agente"])

# ============================================================
# ABA 1: UPLOAD
# ============================================================
with tab1:
    st.header("📤 Envie os arquivos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extrato Bancario")
        st.caption("Aceita CSV ou PDF")
        extrato_file = st.file_uploader("Extrato bancario", type=["csv", "pdf"], key="extrato")

    with col2:
        st.subheader("Livro Razao")
        st.caption("Somente CSV")
        razao_file = st.file_uploader("Livro razao", type=["csv"], key="razao")

    if extrato_file and razao_file:
        # Mostra info sobre o tipo de arquivo
        ext_name = extrato_file.name
        if ext_name.lower().endswith(".pdf"):
            st.info("📄 Extrato em PDF detectado. O sistema vai extrair o texto e tentar identificar os lancamentos.")

        if st.button("Executar Conciliacao", type="primary", use_container_width=True):
            with st.spinner("Processando arquivos..."):
                try:
                    # Salva arquivos temporariamente
                    extrato_path = "/tmp/extrato" + os.path.splitext(extrato_file.name)[1]
                    razao_path = "/tmp/razao.csv"
                    with open(extrato_path, "wb") as f:
                        f.write(extrato_file.getvalue())
                    with open(razao_path, "wb") as f:
                        f.write(razao_file.getvalue())

                    # Carrega dados
                    df_ext, docs_ext = load_extrato_bancario(extrato_path)
                    df_raz, docs_raz = load_livro_razao(razao_path)

                    # Conciliacao
                    df_conc, nao_conc_ext, nao_conc_raz, diverg, resumo = conciliar(
                        df_ext, df_raz,
                        tolerancia_dias=tolerancia_dias,
                        tolerancia_valor=tolerancia_valor,
                        similaridade_minima=similaridade
                    )

                    # Junta todos os documentos para o RAG
                    todos_docs = docs_ext + docs_raz
                    st.session_state.docs = todos_docs
                    st.session_state.resumo = resumo

                    # Constroi vector store
                    build_vector_store(todos_docs)

                    # Salva resultados no estado
                    st.session_state["df_conc"] = df_conc
                    st.session_state["nao_conc_ext"] = nao_conc_ext
                    st.session_state["nao_conc_raz"] = nao_conc_raz
                    st.session_state["diverg"] = diverg
                    st.session_state["resumo"] = resumo
                    st.session_state["df_ext"] = df_ext
                    st.session_state["df_raz"] = df_raz

                    st.success("Conciliacao realizada com sucesso!")
                    st.info(f"Taxa de conciliacao: {len(df_conc)/len(df_ext)*100:.1f}% | Documentos indexados: {len(todos_docs)}")

                except Exception as e:
                    st.error(f"Erro durante a conciliacao: {e}")
    else:
        st.info("Faca o upload dos dois arquivos para iniciar.")

# ============================================================
# ABA 2: RESULTADOS
# ============================================================
with tab2:
    st.header("📊 Resultados da Conciliacao")

    if "df_conc" not in st.session_state:
        st.info("Nenhuma conciliacao realizada ainda. Va para a aba Upload.")
    else:
        # Resumo
        st.subheader("📋 Resumo")
        st.text(st.session_state["resumo"])

        # Conciliados
        st.subheader("Lancamentos Conciliados")
        if not st.session_state["df_conc"].empty:
            st.dataframe(st.session_state["df_conc"], use_container_width=True)
        else:
            st.warning("Nenhum lancamento conciliado.")

        # Nao conciliados
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Nao Conciliados - Extrato")
            if not st.session_state["nao_conc_ext"].empty:
                st.dataframe(st.session_state["nao_conc_ext"], use_container_width=True)
            else:
                st.success("Todos os lancamentos do extrato foram conciliados!")

        with col2:
            st.subheader("Nao Conciliados - Razao")
            if not st.session_state["nao_conc_raz"].empty:
                st.dataframe(st.session_state["nao_conc_raz"], use_container_width=True)
            else:
                st.success("Todos os lancamentos do razao foram conciliados!")

        # Divergencias
        st.subheader("Divergencias Detectadas")
        if not st.session_state["diverg"].empty:
            st.dataframe(st.session_state["diverg"], use_container_width=True)
        else:
            st.success("Nenhuma divergencia detectada!")

# ============================================================
# ABA 3: CHAT COM AGENTE
# ============================================================
with tab3:
    st.header("💬 Perguntar ao Agente de IA")

    if "df_conc" not in st.session_state:
        st.info("Nenhum documento carregado. Va para a aba Upload e execute a conciliacao primeiro.")
    else:
        # Exibe historico
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input do usuario
        pergunta = st.chat_input("Digite sua pergunta sobre a conciliacao...")

        if pergunta:
            # Adiciona pergunta ao historico
            st.session_state.chat_history.append({"role": "user", "content": pergunta})

            with st.chat_message("user"):
                st.markdown(pergunta)

            with st.chat_message("assistant"):
                with st.spinner("Consultando os documentos..."):
                    try:
                        resposta = ask_question(pergunta, st.session_state.get("resumo", ""))
                        texto = resposta["answer"]

                        # Trata resposta em formato de lista (bug do Gemini)
                        if isinstance(texto, list):
                            partes = [item.get("text", "") for item in texto if isinstance(item, dict) and "text" in item]
                            texto = "".join(partes) if partes else str(texto)

                        st.markdown(texto)

                        # Adiciona resposta ao historico
                        st.session_state.chat_history.append({"role": "assistant", "content": texto})

                    except Exception as e:
                        erro = f"Erro ao consultar o agente: {e}"
                        st.error(erro)
                        st.session_state.chat_history.append({"role": "assistant", "content": erro})
