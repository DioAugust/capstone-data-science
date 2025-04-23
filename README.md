# Scholar Dashboard

An end-to-end Semantic Search and Visualization platform for scientific literature. It ingests article metadata from CrossRef, downloads PDFs or HTML, cleans and preprocesses text, generates embeddings with Sentence Transformers, stores them, and provides a Streamlit-based UI for semantic search and cluster visualization.

---

## Features

- **Article Link Extraction**: Queries CrossRef API for a bibliographic search term, filters for full-text availability, and saves article titles, DOIs, and URLs.
- **Article Download & Processing**: Downloads each article (PDF or HTML), extracts text (PyPDF2 for PDF, BeautifulSoup for HTML), cleans and normalizes whitespace.
- **Text Cleaning**: Removes GitHub menus, non-ASCII characters, and extra whitespace; preserves only meaningful content.
- **Embedding Generation**: Converts cleaned document text into 768‑dimensional vectors using a pretrained `sentence-transformers/all-mpnet-base-v2` model, saved as a pickle file.
- **Semantic Search**: Loads embeddings, encodes user queries into vectors, computes cosine similarity, and returns top‑N most relevant documents.
- **Cluster Visualization**: Reduces dimensionality (PCA, t‑SNE, or UMAP), performs K‑Means clustering, and renders scatter plots with convex hulls and keywords.
- **Streamlit Dashboard**:
  - **Search Tab**: Enter natural language queries, view ranked results, and read selected documents with optional term highlighting.
  - **Visualization Tab**: Choose between scatter or clustered views, select reduction method, and adjust cluster count interactively.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` for dependency installation

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/scholar-dashboard.git
   cd scholar-dashboard
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Generate embeddings** (optional if you want fresh embeddings):
   ```bash
   python src/processing/generate_embeddings.py
   ```

### Running the Pipeline

The `main.py` script orchestrates the full pipeline:

```bash
python main.py
```

It will:

1. Extract article links via CrossRef
2. Download and extract text
3. Clean and preprocess articles
4. Generate and save embeddings
5. Launch the Streamlit dashboard

### Launching the Dashboard

If embeddings already exist, you can directly start:

```bash
streamlit run app/dashboard.py --server.fileWatcherType none
```

---

## Project Structure

```
├── data/
│   ├── raw/                # raw JSON links and downloaded PDFs/TXT
│   │   ├── article_links.json
│   │   └── articles/
│   └── processed/          # cleaned article text files
├── embeddings/             # pickled embeddings
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
│       └── embedding_visualizer.py
├── app/
│   └── dashboard.py        # Streamlit app
├── main.py                 # Pipeline orchestration
├── requirements.txt
└── README.md
```

---

## Dependencies

- `requests` and `bs4` for HTTP and HTML parsing
- `PyPDF2` for PDF text extraction
- `crossrefapi` for metadata retrieval
- `sentence-transformers` & `torch` for embeddings
- `scikit-learn` for clustering, dimensionality reduction, and similarity metrics
- `umap-learn` for UMAP
- `streamlit` for interactive UI

List in `requirements.txt`:

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

## Models

- **Embedding Model**: `all-mpnet-base-v2` from Sentence Transformers—balance of speed and semantic accuracy.
- **Clustering**: K‑Means for fixed clusters, with optional dimensionality reduction (PCA or UMAP).
- **Visualization**: Matplotlib-based scatter plots, convex hulls, and annotations.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## License

MIT License © Your Name
