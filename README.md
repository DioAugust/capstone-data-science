# Painel Scholar

Uma plataforma completa de Busca Semântica e Visualização para literatura científica. Ela:

1. Extrai metadados de artigos via CrossRef  
2. Faz download em PDF ou HTML  
3. Limpa e pré-processa o texto  
4. Gera embeddings com Sentence Transformers  
5. Armazena vetores em banco vetorial  
6. Disponibiliza UI em Streamlit para busca semântica e visualização de clusters

---

## Funcionalidades

- **Extração de Links de Artigos**  
  Consulta a API CrossRef com termos bibliográficos, filtra apenas itens com texto completo e salva título, DOI e URL.  
- **Download & Processamento de Artigos**  
  Baixa cada artigo (PDF ou HTML), extrai texto via PyPDF2 (PDF) ou BeautifulSoup (HTML) e normaliza espaços.  
- **Limpeza de Texto**  
  Remove menus do GitHub, caracteres não-ASCII e excesso de espaços, preservando apenas o conteúdo relevante.  
- **Geração de Embeddings**  
  Converte texto limpo em vetores de 768 dimensões usando o modelo pré-treinado `sentence-transformers/all-mpnet-base-v2`, armazenados em arquivo pickle.  
- **Busca Semântica**  
  Carrega embeddings, codifica consultas do usuário em vetores, calcula similaridade de cosseno e retorna os N artigos mais relevantes.  
- **Visualização de Clusters**  
  Reduz dimensionalidade (PCA, t-SNE ou UMAP), aplica K-Means e gera gráficos de dispersão com invólucros convexos e palavras-chave.  
- **Dashboard em Streamlit**  
  - **Aba de Busca**: insira consultas em linguagem natural, veja resultados ordenados e leia artigos com realce opcional de termos.  
  - **Aba de Visualização**: escolha visualização simples ou por clusters, selecione método de redução e ajuste número de clusters interativamente.

---

## Como Começar

### Pré-requisitos

- Python 3.10 ou superior  
- `pip` para instalar dependências  

### Instalação

1. **Clone o repositório**  
   ```bash
   git clone https://github.com/seuusuario/scholar-dashboard.git
   cd scholar-dashboard
   ```

2. **Crie um ambiente virtual**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale dependências**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Opcional) Gere embeddings frescos**  
   ```bash
   python src/processing/generate_embeddings.py
   ```

### Executando o Pipeline

Use o script principal para orquestrar todas as etapas:

```bash
python main.py
```

Ele irá:  
1. Extrair links de artigos via CrossRef  
2. Fazer download e extrair texto  
3. Limpar e pré-processar  
4. Gerar e salvar embeddings  
5. Iniciar o dashboard Streamlit  

### Iniciando o Dashboard

Se já tiver embeddings, execute diretamente:

```bash
streamlit run app/dashboard.py --server.fileWatcherType none
```

---

## Estrutura do Projeto

```
├── data/
│   ├── raw/                
│   │   ├── article_links.json
│   │   └── articles/      
│   └── processed/         
├── embeddings/            
│   └── document_embeddings.pkl
├── src/
│   ├── scraping/
│   │   ├── extract_articles.py
│   │   └── download_articles.py
│   ├── processing/
│   │   ├── generate_embeddings.py
│   │   └── text_cleaning.py
│   ├── search/
│   │   └── semantic_search.py
│   └── visualization/
│       └── cluster_viz.py
├── app/
│   └── dashboard.py       
├── main.py                
├── requirements.txt       
└── README.md              
```

---

## Dependências

- `requests`, `beautifulsoup4` para HTTP e parsing HTML  
- `PyPDF2` para extração de texto em PDF  
- `crossrefapi` para consulta a metadados  
- `sentence-transformers` e `torch` para geração de embeddings  
- `scikit-learn` para clusterização, redução de dimensionalidade e similaridade  
- `umap-learn` para UMAP  
- `streamlit` para interface interativa  

**requirements.txt** inclui:
```
requests
beautifulsoup4
PyPDF2
crossrefapi
sentence-transformers
torch
scikit-learn
umap-learn
streamlit
```

---

## Modelos Utilizados

- **Embedding**: `all-mpnet-base-v2` (Sentence Transformers) — bom equilíbrio entre velocidade e precisão semântica.  
- **Clusterização**: K-Means para grupos fixos, combinado com redução de dimensionalidade (PCA ou UMAP).  
- **Visualização**: gráficos Matplotlib com dispersão, invólucros convexos e anotações de palavras-chave.

---

## Como Contribuir

1. Fork este repositório  
2. Crie uma branch de feature (`git checkout -b feature/nome-da-feature`)  
3. Faça commit das suas mudanças  
4. Abra um Pull Request  

Agradecemos suas contribuições!
