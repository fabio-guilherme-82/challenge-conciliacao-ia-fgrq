# Alura Agent — Conciliação Bancária com IA

> **Challenge Alura ONE G10 — Agente de IA**  
> Autor: Fábio Guilherme (FGRQ)

---

## 📋 Descrição do Projeto

O **Alura Agent — Conciliação Bancária** é um agente de inteligência artificial especializado em contabilidade que automatiza o processo de **conciliação bancária**. A aplicação compara o **extrato bancário** (fornecido pelo banco) com o **livro razão** (lançamentos contábeis da empresa), identifica lançamentos correspondentes, detecta divergências e responde perguntas em **linguagem natural** sobre os dados.

Este projeto foi desenvolvido como solução prática para o desafio do programa **Alura-Oracle ONE G10**, adaptando o conceito de agente RAG (Retrieval-Augmented Generation) para uma necessidade real do dia a dia contábil.

---

## 🏗️ Arquitetura da Solução

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Extrato Banc.  │     │   Livro Razão    │     │   Usuário       │
│    (CSV)        │     │     (CSV)        │     │  (Perguntas)    │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DOCUMENT LOADER + PARSER                      │
│  • Normalização de colunas (data, valor, descrição, tipo)        │
│  • Conversão de cada linha em documento vetorial                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CONCILIADOR (REGRAS DE NEGÓCIO)                │
│  • Matching por: valor, data (±dias), descrição (similaridade)   │
│  • Identificação de lançamentos não conciliados                  │
│  • Detecção de divergências (diferença de data/valor)            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              VECTOR STORE (ChromaDB + SentenceTransformers)       │
│  • Embeddings: all-MiniLM-L6-v2                                 │
│  • Armazenamento semântico dos documentos                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              AGENTE RAG — LLM (Google Gemini 1.5 Flash)          │
│  • Recuperação de contexto relevante                             │
│  • Geração de respostas em português com valores em R$           │
│  • Respostas baseadas apenas nos documentos carregados           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

| Camada | Tecnologia | Finalidade |
|--------|-----------|------------|
| **Interface** | Streamlit | Aplicação web interativa |
| **Processamento** | Pandas | Manipulação e normalização de dados |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Vetorização semântica do texto |
| **Vector DB** | ChromaDB | Armazenamento e busca vetorial |
| **LLM** | Google Gemini 1.5 Flash | Geração de respostas naturais |
| **Framework IA** | LangChain (LCEL) | Orquestração do pipeline RAG |
| **Leitura PDF** | PyPDF | Extração de texto de PDFs |
| **Deploy** | Streamlit Community Cloud | Hospedagem na nuvem |

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior
- Chave de API do Google Gemini (obtenha em [aistudio.google.com](https://aistudio.google.com))

### Passo a passo

1. **Clone o repositório:**
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

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a API Key:**

   Crie um arquivo `.env` na raiz do projeto:
   ```env
   GOOGLE_API_KEY=sua_chave_aqui
   ```

   Ou configure no Streamlit Cloud em **Settings → Secrets**:
   ```toml
   GOOGLE_API_KEY = "sua_chave_aqui"
   ```

5. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

6. **Acesse no navegador:**
   ```
   http://localhost:8501
   ```

---

## 📤 Estrutura dos Arquivos de Entrada

### Extrato Bancário (CSV)
O arquivo deve conter as colunas (nomes flexíveis):
- `data` / `date` / `data_mov` — Data da movimentação
- `descricao` / `historico` / `desc` — Descrição do lançamento
- `valor` / `vlr` / `amount` — Valor em R$
- `tipo` / `natureza` / `dc` — Débito (D) ou Crédito (C)
- `saldo` / `balance` — Saldo acumulado (opcional)

### Livro Razão (CSV)
O arquivo deve conter as colunas (nomes flexíveis):
- `data` / `date` / `data_lancamento` — Data do lançamento contábil
- `descricao` / `historico` — Histórico do lançamento
- `valor` / `vlr` — Valor em R$
- `conta` / `conta_contabil` — Código da conta contábil
- `documento` / `doc` / `ndoc` — Número do documento
- `tipo` / `dc` — Débito (D) ou Crédito (C)

---

## 💬 Exemplos de Perguntas que o Agente Responde

> Após fazer o upload do extrato e do livro razão, você pode conversar com o agente na aba **"💬 Perguntar ao Agente"**.

### Sobre conciliação
- *"Qual a taxa de conciliação deste mês?"*
- *"Quais lançamentos do extrato não foram conciliados?"*
- *"Quais lançamentos do razão não aparecem no extrato?"*

### Sobre valores e saldos
- *"Qual o saldo final do extrato bancário?"*
- *"Qual o total de tarifas bancárias no período?"*
- *"Qual o valor total dos lançamentos de crédito?"*

### Sobre divergências
- *"Existem lançamentos conciliados com diferença de data?"*
- *"Há divergências de valor entre extrato e razão?"*
- *"Quais lançamentos têm descrições similares mas valores diferentes?"*

### Sobre análise contábil
- *"Resuma os principais pontos de atenção desta conciliação."*
- *"O extrato está equilibrado com o razão?"*
- *"Quais documentos do razão estão pendentes de compensação bancária?"*

---

## 📝 Exemplos de Respostas Geradas pelo Agente

### Exemplo 1 — Taxa de conciliação
> **Pergunta:** *Qual a taxa de conciliação?*
> 
> **Resposta:** A taxa de conciliação entre o extrato bancário e o livro razão é de **87,5%**. Foram encontrados 35 lançamentos conciliados de um total de 40 no extrato. Há 5 lançamentos no extrato e 3 no razão sem correspondência.

### Exemplo 2 — Não conciliados
> **Pergunta:** *Quais lançamentos do extrato não foram conciliados?*
> 
> **Resposta:** Os seguintes lançamentos do extrato não possuem correspondência no livro razão:
> - 15/08/2026 | TARIFA BANCÁRIA MENSAL | R$ 45,90 | DÉBITO
> - 18/08/2026 | TRANSFERÊNCIA PIX - FORNECEDOR X | R$ 1.250,00 | DÉBITO
> - 22/08/2026 | DEPÓSITO EM DINHEIRO | R$ 500,00 | CRÉDITO
> 
> Recomendo verificar se os lançamentos contábeis foram registrados com datas ou descrições diferentes.

### Exemplo 3 — Divergências
> **Pergunta:** *Há divergências de valor?*
> 
> **Resposta:** Sim, identifiquei **2 ocorrências** de divergência de valor nos lançamentos conciliados:
> - PAGAMENTO DE ENERGIA ELÉTRICA: extrato R$ 1.234,56 vs razão R$ 1.230,00 (diferença de R$ 4,56)
> - REEMBOLSO DE DESPESAS: extrato R$ 350,00 vs razão R$ 35,00 (diferença de R$ 315,00 — possível erro de digitação)

---

## 📂 Estrutura do Repositório

```
challenge-conciliacao-ia-fgrq/
├── app.py                 # Interface Streamlit
├── agent.py               # Agente RAG (LangChain + Fallback)
├── document_loader.py     # Carrega e normaliza CSV/PDF
├── conciliador.py         # Lógica de matching e divergências
├── requirements.txt       # Dependências do projeto
├── README.md              # Este arquivo
├── .env.example           # Exemplo de variáveis de ambiente
├── .streamlit/            # Configurações do Streamlit
│   └── config.toml
├── data/                  # Dados de exemplo (opcional)
└── deploy_oci.sh          # Script de deploy na OCI
```

---

## ☁️ Deploy na Nuvem

A aplicação está implantada e funcionando no **Streamlit Community Cloud**.

🔗 **Link público:** [challenge-conciliacao-ia-fgrq-tvcazycfy6ioyq8dtglgjx.streamlit.app](https://challenge-conciliacao-ia-fgrq-tvcazycfy6ioyq8dtglgjx.streamlit.app)

### Como fazer o deploy
1. Conecte seu repositório GitHub ao [Streamlit Cloud](https://streamlit.io/cloud)
2. Selecione o repositório `challenge-conciliacao-ia-fgrq`
3. Configure a secret `GOOGLE_API_KEY` em **Settings → Secrets**
4. Clique em **Deploy**

---

## ⚙️ Configurações de Conciliação

Na barra lateral da aplicação, você pode ajustar:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| **Tolerância de dias** | 3 | Diferença máxima de data entre extrato e razão |
| **Tolerância de valor** | R$ 0,01 | Diferença máxima de valor aceitável |
| **Similaridade mínima** | 0,4 | Similaridade textual mínima entre descrições (0 a 1) |

---

## 🧪 Testes e Validação

O sistema foi testado com:
- Extratos bancários reais (formato CSV)
- Livros razão de pequenas e médias empresas
- PDFs de demonstrativos bancários
- Diferentes padrões de nomenclatura de colunas

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais no programa **Alura-Oracle ONE G10**.

---

## 👤 Autor

**Fábio Guilherme** — [GitHub](https://github.com/fabio-guilherme-82)

> *"Automatizando a contabilidade com inteligência artificial."*
