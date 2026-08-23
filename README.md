# 🏦 Alura Agent — Conciliacao Bancaria com IA

> **Challenge Oracle + Alura (ONE AI For Tech) — G10 Brasil**

O **Alura Agent** e um agente de inteligencia artificial especializado em **conciliacao bancaria**. Ele compara automaticamente o **extrato bancario** (fornecido pelo banco) com o **livro razao** (lancamentos contabeis da empresa), identifica correspondencias, detecta divergencias e responde perguntas em **linguagem natural** sobre os dados financeiros.

Este projeto foi desenvolvido como solucao pratica para o desafio do programa **Alura-Oracle ONE G10**, aplicando o conceito de **RAG (Retrieval-Augmented Generation)** a um problema real do dia a dia contabil.

---

## 🌐 Aplicacao Online (Deploy)

Acesse a aplicacao em funcionamento:
👉 **[Alura Agent — Streamlit Community Cloud](https://challenge-conciliacao-ia-fgrq-tvcazycfy6ioyq8dtglgjx.streamlit.app)**

---

## 📝 Descricao do Projeto

O Alura Agent e um sistema **RAG (Retrieval-Augmented Generation)** que combina:

1. **Ingestao de dados:** Leitura de arquivos CSV de extrato bancario e livro razao
2. **Normalizacao:** Padronizacao automatica de colunas (data, valor, descricao, tipo, etc.)
3. **Conciliacao:** Algoritmo de matching que compara lancamentos por valor, data (com tolerancia) e similaridade textual
4. **Vetorizacao:** Conversao dos dados em embeddings semanticos via Google Gemini
5. **Consulta inteligente:** Respostas geradas pelo LLM Gemini 1.5 Flash baseadas exclusivamente nos documentos carregados

O objetivo e reduzir o tempo gasto na conciliacao manual, identificar erros contabeis e fornecer insights financeiros via chat em linguagem natural.

---

## 🏛️ Arquitetura da Solucao

```
[Extrato Bancario CSV] ──┐
                         ├──> [Document Loader] ──> [Normalizacao de Dados]
[Livro Razao CSV] ───────┘                              │
                                                        ▼
                                              [Conciliador (Regras de Negocio)]
                                                        │
              [Interface Streamlit] <── [LLM Gemini 1.5 Flash] <── [Contexto] <── [ChromaDB VectorStore]
                                                        ▲
                                              [Google Gemini Embeddings]
```

### Fluxo de dados:
1. **Upload:** O usuario envia dois arquivos CSV (extrato e razao)
2. **Parsing:** Cada linha e convertida em um documento estruturado
3. **Conciliacao:** O algoritmo cruza lancamentos por valor, data e descricao
4. **Chunking:** Os documentos sao fragmentados em chunks de 1000 caracteres
5. **Embeddings:** Google `gemini-embedding-001` converte texto em vetores semanticos
6. **Armazenamento:** ChromaDB indexa os vetores para busca rapida
7. **Resposta:** O modelo Gemini responde perguntas baseadas apenas nos documentos carregados

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

| Camada | Tecnologia | Finalidade |
|--------|-----------|------------|
| **Interface** | Streamlit | Aplicacao web interativa |
| **Processamento** | Pandas | Manipulacao e normalizacao de dados |
| **Embeddings** | Google Gemini Embedding API | Vetorizacao semantica do texto |
| **Vector DB** | ChromaDB | Armazenamento e busca vetorial |
| **LLM** | Google Gemini 1.5 Flash | Geracao de respostas naturais |
| **Framework IA** | LangChain (LCEL) | Orquestracao do pipeline RAG |
| **Leitura CSV** | Pandas | Parsing de arquivos CSV |
| **Hospedagem** | Streamlit Community Cloud | Deploy na nuvem |

---

## 🚀 Como Executar o Projeto

### Prerequisitos
- Python 3.10 ou superior
- Chave de API do Google Gemini (obtenha gratuitamente em [aistudio.google.com](https://aistudio.google.com))

### Passo a passo

1. **Clone o repositorio:**
   ```bash
   git clone https://github.com/fabio-guilherme-82/challenge-conciliacao-ia-fgrq.git
   cd challenge-conciliacao-ia-fgrq
   ```

2. **Crie o ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```

3. **Instale as dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a API Key:**

   Copie o arquivo de exemplo:
   ```bash
   cp .env.example .env
   ```

   Edite o arquivo `.env` e insira sua chave:
   ```env
   GOOGLE_API_KEY=sua_chave_aqui
   ```

   > **Para deploy no Streamlit Cloud:** Configure a secret em **Settings → Secrets**:
   > ```toml
   > GOOGLE_API_KEY = "sua_chave_aqui"
   > ```

5. **Execute a aplicacao:**
   ```bash
   streamlit run app.py
   ```

6. **Acesse no navegador:**
   ```
   http://localhost:8501
   ```

---

## 📤 Estrutura dos Arquivos de Entrada

### Extrato Bancario (CSV)
O arquivo deve conter as colunas (nomes flexiveis, o sistema detecta automaticamente):

| Coluna esperada | Variacoes aceitas |
|-----------------|-------------------|
| `data` | data, date, dt, data_mov, data_transacao |
| `descricao` | descricao, historico, desc, detalhe |
| `valor` | valor, vlr, amount, val |
| `tipo` | tipo, natureza, operacao, debito_credito, dc |
| `saldo` | saldo, saldo_atual, balance |

**Exemplo de linha:**
```csv
data,descricao,valor,tipo,saldo
01/08/2026,SALDO ANTERIOR,0.00,,15000.00
02/08/2026,PAGAMENTO FORNECEDOR ABC,-2500.00,D,12500.00
03/08/2026,RECEBIMENTO CLIENTE XYZ,3200.00,C,15700.00
```

### Livro Razao (CSV)
O arquivo deve conter as colunas (nomes flexiveis):

| Coluna esperada | Variacoes aceitas |
|-----------------|-------------------|
| `data` | data, date, data_lancamento |
| `descricao` | descricao, historico |
| `valor` | valor, vlr |
| `conta` | conta, conta_contabil, codigo_conta |
| `documento` | documento, doc, numero_documento |
| `tipo` | tipo, dc, debito_credito |

**Exemplo de linha:**
```csv
data,descricao,valor,conta,documento,tipo
02/08/2026,PAGTO FORNECEDOR ABC,2500.00,21101,001234,D
03/08/2026,REC CLIENTE XYZ,3200.00,11201,005678,C
```

---

## 💬 Exemplos de Perguntas que o Agente Responde

> Apos fazer o upload do extrato e do livro razao, voce pode conversar com o agente na aba **"💬 Perguntar ao Agente"**.

### Sobre conciliacao
- *"Qual a taxa de conciliacao deste mes?"*
- *"Quais lancamentos do extrato nao foram conciliados?"*
- *"Quais lancamentos do razao nao aparecem no extrato?"*

### Sobre valores e saldos
- *"Qual o saldo final do extrato bancario?"*
- *"Qual o total de tarifas bancarias no periodo?"*
- *"Qual o valor total dos lancamentos de credito?"*

### Sobre divergencias
- *"Existem lancamentos conciliados com diferenca de data?"*
- *"Ha divergencias de valor entre extrato e razao?"*
- *"Quais lancamentos tem descricoes similares mas valores diferentes?"*

### Sobre analise contabil
- *"Resuma os principais pontos de atencao desta conciliacao."*
- *"O extrato esta equilibrado com o razao?"*
- *"Quais documentos do razao estao pendentes de compensacao bancaria?"*

---

## 📝 Exemplos de Respostas Geradas pelo Agente

### Exemplo 1 — Taxa de conciliacao
> **Pergunta:** *Qual a taxa de conciliacao?*
>
> **Resposta:** A taxa de conciliacao entre o extrato bancario e o livro razao e de **87,5%**. Foram encontrados 35 lancamentos conciliados de um total de 40 no extrato. Ha 5 lancamentos no extrato e 3 no razao sem correspondencia.

### Exemplo 2 — Nao conciliados
> **Pergunta:** *Quais lancamentos do extrato nao foram conciliados?*
>
> **Resposta:** Os seguintes lancamentos do extrato nao possuem correspondencia no livro razao:
> - 15/08/2026 | TARIFA BANCARIA MENSAL | R$ 45,90 | DEBITO
> - 18/08/2026 | TRANSFERENCIA PIX - FORNECEDOR X | R$ 1.250,00 | DEBITO
> - 22/08/2026 | DEPOSITO EM DINHEIRO | R$ 500,00 | CREDITO
>
> Recomendo verificar se os lancamentos contabeis foram registrados com datas ou descricoes diferentes.

### Exemplo 3 — Divergencias
> **Pergunta:** *Ha divergencias de valor?*
>
> **Resposta:** Sim, identifiquei **2 ocorrencias** de divergencia de valor nos lancamentos conciliados:
> - PAGAMENTO DE ENERGIA ELETRICA: extrato R$ 1.234,56 vs razao R$ 1.230,00 (diferenca de R$ 4,56)
> - REEMBOLSO DE DESPESAS: extrato R$ 350,00 vs razao R$ 35,00 (diferenca de R$ 315,00 — possivel erro de digitacao)

---

## 📂 Estrutura do Repositorio

```
challenge-conciliacao-ia-fgrq/
├── app.py                 # Interface Streamlit
├── agent.py               # Motor RAG (LangChain + Google Gemini)
├── document_loader.py     # Carrega e normaliza CSVs
├── conciliador.py         # Logica de matching e divergencias
├── requirements.txt       # Dependencias do projeto
├── README.md              # Documentacao completa
├── .env.example           # Exemplo de variaveis de ambiente
└── .streamlit/            # Configuracoes do Streamlit (opcional)
    └── config.toml
```

---

## ☁️ Deploy na Nuvem

A aplicacao esta implantada no **Streamlit Community Cloud**.

🔗 **Link publico:** [challenge-conciliacao-ia-fgrq-tvcazycfy6ioyq8dtglgjx.streamlit.app](https://challenge-conciliacao-ia-fgrq-tvcazycfy6ioyq8dtglgjx.streamlit.app)

### Como fazer o deploy
1. Conecte seu repositorio GitHub ao [Streamlit Cloud](https://streamlit.io/cloud)
2. Selecione o repositorio `challenge-conciliacao-ia-fgrq`
3. Configure a secret `GOOGLE_API_KEY` em **Settings → Secrets**
4. Clique em **Deploy**

---

## 🔧 Configuracoes de Conciliacao

Na barra lateral da aplicacao, voce pode ajustar os parametros do algoritmo:

| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| **Tolerancia de dias** | 3 | Diferenca maxima de data entre extrato e razao |
| **Tolerancia de valor** | R$ 0,01 | Diferenca maxima de valor aceitavel |
| **Similaridade minima** | 0,4 | Similaridade textual minima entre descricoes (0 a 1) |

---

## 🧪 Testes e Validacao

O sistema foi testado com:
- Extratos bancarios reais (formato CSV)
- Livros razao de pequenas e medias empresas
- Diferentes padroes de nomenclatura de colunas
- Valores com formatacao brasileira (R$, virgula decimal)

---

## 📄 Licenca

Este projeto foi desenvolvido para fins educacionais no programa **Alura-Oracle ONE G10**.

---

## 👤 Autor

**Fabio Guilherme** — [GitHub](https://github.com/fabio-guilherme-82)

> *"Automatizando a contabilidade com inteligencia artificial."*
