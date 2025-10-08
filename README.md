# 🧠 AI Fitness Coach – Projeto de Agentes de IA para Mestrado SAEG/IFF

## 📘 Descrição Geral

Este projeto foi desenvolvido como parte da disciplina de **Inteligência Artificial** do **Mestrado Profissional em Sistemas Aplicados à Engenharia e Gestão (SAEG)** do **Instituto Federal Fluminense (IFF)**.  
O objetivo é demonstrar a aplicação prática de **agentes de IA** para gerar **planos de treino personalizados** com base em informações fornecidas pelo usuário.

O sistema utiliza conceitos de **RAG (Retrieval-Augmented Generation)**, empregando como base de conhecimento um **livro, apostila ou material textual sobre musculação e força**, que **deve ser inserido pelo usuário** na pasta `treino/`.  

> ⚠️ **Importante:** O livro *Enciclopédia de Musculação e Força* de **Jim Stoppani**, utilizado neste experimento acadêmico, **não está incluído neste repositório** devido a questões de licenciamento.  
> O usuário pode adicionar **qualquer outro material próprio ou de uso livre** (em PDF ou TXT) e ajustar o nome no código para usá-lo como fonte RAG.

A aplicação é executada em **Streamlit**, permitindo interação direta com o usuário por meio de uma interface web intuitiva.

---

## 🧩 Estrutura do Projeto

```
📁 AI_Fitness_Coach/
│
├── 📓 AI_Fitness_Coach.ipynb       # Notebook base com experimentos e arquitetura original
├── 🧠 app.py                        # Aplicativo principal em Streamlit
├── ⚙️ requirements.txt              # Dependências e bibliotecas necessárias
├── 🌱 .env_example                  # Variáveis de ambiente de exemplo
└── 📚 treino/
    └── (adicione aqui seu arquivo de base RAG, ex: musculacao_base.pdf)
```

---

## 🚀 Tecnologias e Conceitos Envolvidos

### 🔧 **Frameworks e Ferramentas**
- **Streamlit** – Interface interativa para o usuário.  
- **Python 3.10+**  
- **LangChain / LangGraph** – Arquitetura de agentes e pipelines de IA (no notebook original).  
- **FAISS / Pinecone** – Indexação vetorial e recuperação semântica.  
- **Sentence Transformers** – Geração de embeddings textuais.  
- **DuckDuckGo API** – Pesquisa web complementar.  
- **dotenv** – Gestão de variáveis sensíveis (.env).  

### 🤖 **Conceitos de Inteligência Artificial Aplicados**
- **Agentes de IA autônomos** para composição de planos de treino.  
- **Raciocínio guiado por objetivos** e heurísticas.  
- **RAG (Retrieval-Augmented Generation)**: combinação de recuperação de conhecimento (documento base) e geração de conteúdo.  
- **Personalização baseada em dados**: idade, peso, frequência, objetivo e experiência do usuário.

---

## ⚙️ Instalação e Execução

### 1. Clonar o Repositório
```bash
git clone https://github.com/andersonpaes/ai_fitness_coach.git
cd AI_Fitness_Coach
```

### 2. Criar Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente
Crie um arquivo `.env` baseado em `.env_example` e preencha com suas chaves de API, se necessário:
```bash
cp .env_example .env
```

### 5. Inserir o Material de Conhecimento (RAG)
Adicione na pasta `treino/` um arquivo PDF ou TXT contendo o conteúdo base de musculação, por exemplo:
```
treino/musculacao_base.pdf
```

E no código (por exemplo, no notebook ou módulo RAG), ajuste o caminho:
```python
rag_path = "treino/musculacao_base.pdf"
```

### 6. Executar o Aplicativo Streamlit
```bash
streamlit run app.py
```

---

## 🧭 Fluxo do Sistema

```mermaid
flowchart TD
    A[Usuário] -->|Entrada de dados (idade, peso, objetivo, etc.)| B[Interface Streamlit]
    B --> C[Normalização dos dados]
    C --> D[Análise matemática (calorias, repetições, volume)]
    D --> E[Consulta RAG]
    E -->|Busca no livro/apostila em treino/| F[Base de conhecimento personalizada]
    D --> G[Pesquisa Web via DuckDuckGo]
    F --> H[Geração do Plano Personalizado]
    G --> H
    H --> I[Renderização em Markdown]
    I --> J[Exibição na Interface]
    J -->|Usuário visualiza plano final| A
```

---

## 📊 Componentes Principais

| Arquivo | Descrição |
|----------|------------|
| `AI_Fitness_Coach.ipynb` | Notebook com experimentos e arquitetura de agentes (LangGraph, Groq, FAISS, etc). |
| `app.py` | Aplicação Streamlit que coleta dados, realiza análises e gera o plano final. |
| `.env_example` | Exemplo de configuração de variáveis de ambiente (ex.: chaves de API). |
| `requirements.txt` | Lista completa de dependências Python. |
| `treino/` | Pasta onde o usuário insere seu material textual (livro, apostila, etc). |

---

## 🧮 Exemplo de Integração RAG (FAISS + LangChain)

```python
# Exemplo de integração RAG com LangChain + FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# 1️⃣ Caminho do material base
rag_path = "treino/musculacao_base.pdf"

# 2️⃣ Carrega o texto do PDF
loader = PyPDFLoader(rag_path)
docs = loader.load()

# 3️⃣ Divide o texto em blocos menores para indexação
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 4️⃣ Gera embeddings (vetores numéricos) usando modelo público
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5️⃣ Cria e salva o índice FAISS
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("faiss_index")

# 6️⃣ Consulta de exemplo
query = "melhores exercícios para hipertrofia de peito"
docs_similares = db.similarity_search(query, k=3)

print("Resultados:")
for d in docs_similares:
    print("-", d.page_content[:300], "...")
```

---

## 👨‍🏫 Autor e Contexto Acadêmico

**Trabalho desenvolvido para:**  
Disciplina de **Inteligência Artificial**  
Programa de Pós-Graduação SAEG – **Instituto Federal Fluminense (IFF)**  

**Autor:** Anderson Paes Gomes  
**Orientador:** Luiz Gustavo 
**Ano:** 2025  
