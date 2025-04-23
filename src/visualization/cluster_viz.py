import os
import pickle
import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import umap

from src.processing.generate_embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingVisualizer:
    """
    Loads (or generates) embeddings and provides methods
    to visualize them via scatter plots and clustered plots.
    """

    def __init__(
        self,
        embeddings_file: str = "embeddings/document_embeddings.pkl",
        processed_docs_folder: str = "data/processed/"
    ):
        self.embeddings_file = embeddings_file
        self.processed_docs_folder = processed_docs_folder
        self.doc_names: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

        # Generate embeddings if missing
        if not os.path.exists(self.embeddings_file):
            logging.warning(f"Embeddings file not found: {self.embeddings_file}. Generating now...")
            generator = EmbeddingGenerator(
                processed_docs_folder=self.processed_docs_folder,
                embeddings_file=self.embeddings_file
            )
            generator.run()
            if not os.path.exists(self.embeddings_file):
                raise FileNotFoundError(f"Failed to generate embeddings at {self.embeddings_file}")

        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load and normalize embeddings from disk."""
        try:
            with open(self.embeddings_file, 'rb') as f:
                data: Dict[str, np.ndarray] = pickle.load(f)
            self.doc_names = list(data.keys())
            emb = np.vstack(list(data.values()))
            self.doc_embeddings = normalize(emb)  # unit‐length rows
            logging.info(f"Loaded and normalized {len(self.doc_names)} embeddings.")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            self.doc_embeddings = None

    def _reduce_dimensionality(
        self,
        embeddings: np.ndarray,
        method: str
    ) -> Optional[np.ndarray]:
        """Reduce embeddings to 2D via PCA, t-SNE, or UMAP."""
        logging.info(f"Reducing dimensionality with {method.upper()}")
        try:
            if method == 'pca':
                return PCA(n_components=2, random_state=42).fit_transform(embeddings)

            if method == 'tsne':
                if embeddings.shape[1] > 50:
                    embeddings = PCA(n_components=50, random_state=42).fit_transform(embeddings)
                return TSNE(n_components=2, random_state=42, init='pca').fit_transform(embeddings)

            if method == 'umap':
                return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)

            logging.error(f"Unknown reduction method: {method}")
            return None
        except Exception as e:
            logging.error(f"Error during dimensionality reduction ({method}): {e}")
            return None

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> Optional[np.ndarray]:
        """Cluster embeddings into n_clusters via KMeans."""
        logging.info(f"Clustering into {n_clusters} clusters with KMeans")
        try:
            return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
        except Exception as e:
            logging.error(f"Error during KMeans clustering: {e}")
            return None

    def _extract_cluster_keywords(
        self,
        labels: np.ndarray,
        n_clusters: int
    ) -> Dict[int, str]:
        """Extract up to 3 representative keywords per cluster from filenames."""
        stop_words = {
            'and','the','for','with','from','that','this','are','not','but','have','has'
        }
        keywords: Dict[int, str] = {}
        for cid in range(n_clusters):
            members = [self.doc_names[i] for i, lab in enumerate(labels) if lab == cid]
            if not members:
                keywords[cid] = f"Cluster {cid} (empty)"
                continue
            tokens = []
            for name in members:
                tokens.extend(name.replace('.txt','').replace('_',' ').split())
            freq: Dict[str,int] = {}
            for tok in tokens:
                tok_l = tok.lower()
                if tok_l.isalpha() and len(tok_l)>2 and tok_l not in stop_words:
                    freq[tok_l] = freq.get(tok_l,0)+1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
            keywords[cid] = ', '.join(w for w,_ in top) if top else members[0]
        return keywords

    def plot_scatter(self, method: str = 'pca') -> Optional[plt.Figure]:
        """Simple 2D scatter of reduced embeddings."""
        if self.doc_embeddings is None:
            logging.error("No embeddings loaded for scatter plot.")
            return None
        reduced = self._reduce_dimensionality(self.doc_embeddings, method)
        if reduced is None:
            return None
        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(reduced[:,0], reduced[:,1], alpha=0.7)
        ax.set_title(f"Embedding Scatter ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()
        return fig

    def plot_clusters(
        self,
        method: str = 'pca',
        n_clusters: int = 5
    ) -> Optional[plt.Figure]:
        """2D scatter colored by cluster with convex‐hull overlays and labels."""
        if self.doc_embeddings is None:
            logging.error("No embeddings loaded for cluster plot.")
            return None
        # First reduce and then cluster on reduced to match visualisation
        reduced = self._reduce_dimensionality(self.doc_embeddings, method)
        if reduced is None:
            return None
        labels = self._cluster_embeddings(reduced, n_clusters)
        if labels is None:
            return None

        keywords = self._extract_cluster_keywords(labels, n_clusters)

        fig, ax = plt.subplots(figsize=(14,10))
        palette = plt.cm.tab10(np.linspace(0,1,n_clusters))

        # Draw convex hulls
        for cid in range(n_clusters):
            pts = reduced[labels==cid]
            if pts.shape[0] >= 3:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.fill(poly[:,0], poly[:,1], color=palette[cid], alpha=0.2)

        # Plot points
        ax.scatter(
            reduced[:,0], reduced[:,1],
            c=[palette[l] for l in labels],
            edgecolor='k', alpha=0.8
        )

        # Annotate cluster centers
        for cid in range(n_clusters):
            pts = reduced[labels==cid]
            if pts.size:
                center = pts.mean(axis=0)
                ax.text(
                    center[0], center[1], keywords[cid],
                    fontsize=12, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, pad=2)
                )

        ax.set_title(f"Clusters ({method.upper()} | k={n_clusters})")
        ax.axis('off')
        plt.tight_layout()
        return fig

    def interactive_console(self) -> None:
        """Basic console UI to choose and display plots."""
        if self.doc_embeddings is None:
            print("Embeddings not available. Generate first.")
            return
        choice = input("1) Scatter plot  2) Cluster plot  (enter 1 or 2): ").strip()
        method = input("Reduction method (pca, tsne, umap): ").strip().lower()
        if choice == '1':
            fig = self.plot_scatter(method)
        else:
            k = int(input("Number of clusters: ").strip() or "5")
            fig = self.plot_clusters(method, k)
        if fig:
            print("Displaying plot. Close window to continue.")
            plt.show()
        else:
            print("Failed to generate plot.")
