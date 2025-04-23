import os
import sys
import re
import streamlit as st
from PIL import Image

# ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cache heavy resources to improve performance
@st.cache_resource
def get_searcher():
    from src.search.semantic_search import SemanticSearcher
    return SemanticSearcher()

@st.cache_resource
def get_visualizer():
    from src.visualization.cluster_viz import EmbeddingVisualizer
    return EmbeddingVisualizer()

# Load header and logo
HEADER_IMG = "data/header.png"
LOGO_IMG = "data/logo.png"

# Page configuration
st.set_page_config(
    page_title="Scholar Dashboard",
    page_icon=LOGO_IMG,
    layout="wide"
)

# Base paths
PROCESSED_DOCS = "data/processed/"

def render_document(name: str, content: str, terms: list, highlight: bool):
    """Render document text with optional term highlighting."""
    safe_name = name.replace('_', ' ').title()
    if highlight and terms:
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        content = re.sub(pattern, r"<mark>\1</mark>", content, flags=re.IGNORECASE)
    st.markdown(f"""
    <div style="color:#000;background:#f9f9f9;padding:16px;border-radius:8px;max-height:600px;overflow:auto;font-family:sans-serif;">
      <h3 style="margin-top:0;">📄 {safe_name}</h3>
      <pre style="white-space:pre-wrap;line-height:1.4;">{content}</pre>
    </div>
    """, unsafe_allow_html=True)

def run_dashboard():
    # Initialize session state
    defaults = {
        'results': [],
        'doc_options': {},
        'current_doc': None,
        'query': ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Sidebar controls
    st.sidebar.image(LOGO_IMG, use_container_width=True)
    st.sidebar.title("Controls")
    st.session_state.query = st.sidebar.text_input(
        "Search articles...",
        value=st.session_state.query,
        key="query_input"
    )
    search_button = st.sidebar.button("Search")
    st.sidebar.markdown("---")
    viz_mode = st.sidebar.radio("Visualization mode", ["None", "Scatter", "Clusters"])
    reduction_method = None
    clusters = None
    if viz_mode != "None":
        reduction_method = st.sidebar.selectbox(
            "Dimensionality reduction",
            [ "UMAP", "PCA"],
            key="reduce_method"
        ).lower()
        if viz_mode == "Clusters":
            max_k = len(st.session_state.results) or 10
            clusters = st.sidebar.slider(
                "Number of clusters",
                min_value=2,
                max_value=min(max_k, 20),
                value=min(6, max_k),
                key="cluster_k"
            )
    st.sidebar.markdown("---")
    highlight = st.sidebar.checkbox("Highlight query terms", key="highlight_terms")
    st.sidebar.write("v1.0")

    # Header
    cols = st.columns([1, 3, 1])
    with cols[1]:
        if os.path.exists(HEADER_IMG):
            st.image(HEADER_IMG, use_container_width=True)
        st.title("Scholar Dashboard")

    # Load modules
    with st.spinner("Loading models and embeddings..."):
        searcher = get_searcher()
        visualizer = get_visualizer()

    if getattr(searcher, 'doc_embeddings', None) is None:
        st.error("Error loading embeddings. Please run the embedding generator first.")
        return

    tab1, tab2 = st.tabs(["🔍 Search", "📊 Visualization"])

    # --- Search Tab ---
    with tab1:
        st.header("Semantic Search")
        if search_button and st.session_state.query:
            with st.spinner("Searching..."):
                results = searcher.search(st.session_state.query) or []
            st.session_state.results = results
            st.session_state.doc_options = {
                f"{name} ({score:.3f})": name for name, score in results
            }
            st.session_state.current_doc = None

        if st.session_state.results:
            st.success(f"{len(st.session_state.results)} documents found")
            choice = st.selectbox(
                "Select document",
                options=list(st.session_state.doc_options.keys()),
                key="doc_selector"
            )
            if choice:
                st.session_state.current_doc = st.session_state.doc_options[choice]

        if st.session_state.current_doc:
            doc_file = st.session_state.current_doc
            path = os.path.join(PROCESSED_DOCS, doc_file)
            if os.path.exists(path):
                text = open(path, 'r', encoding='utf-8').read()
                render_document(
                    name=doc_file,
                    content=text,
                    terms=st.session_state.query.split() if highlight else [],
                    highlight=highlight
                )
            else:
                st.error("File not found.")
        elif st.session_state.query and not st.session_state.results:
            st.warning("No documents found.")

    # --- Visualization Tab ---
    with tab2:
        st.header("Embedding Visualization")
        if viz_mode == "None":
            st.info("Select a visualization mode from the sidebar.")
        else:
            if viz_mode == "Scatter":
                fig = visualizer.plot_scatter(reduction_method)
            else:  # Clusters
                fig = visualizer.plot_clusters(reduction_method, clusters)
            if fig:
                st.pyplot(fig)
            else:
                st.error("Visualization error. Check logs.")

if __name__ == '__main__':
    run_dashboard()
