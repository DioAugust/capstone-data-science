# src/visualization/embedding_visualizer.py

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import umap

from src.processing.generate_embeddings import EmbeddingGenerator

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingVisualizer:
    """
    Classe para carregar (ou gerar) embeddings e gerar visualizações de clusters.
    """

    def __init__(self,
                 embeddings_file: str = "embeddings/document_embeddings.pkl",
                 processed_docs_folder: str = "data/processed/"):
        """
        Inicializa o EmbeddingVisualizer.

        Se o arquivo de embeddings não existir, ele será gerado automaticamente.
        """
        self.embeddings_file = embeddings_file
        self.processed_docs_folder = processed_docs_folder
        self.doc_names = None
        self.doc_embeddings = None

        # Gera embeddings se não existir
        if not os.path.exists(self.embeddings_file):
            logging.warning(f"Arquivo de embeddings não encontrado: {self.embeddings_file}. Gerando agora...")
            gen = EmbeddingGenerator(
                processed_docs_folder=self.processed_docs_folder,
                embeddings_file=self.embeddings_file
            )
            gen.run()
            if not os.path.exists(self.embeddings_file):
                logging.error("Falha ao gerar embeddings. Verifique o pipeline anterior.")
                raise FileNotFoundError(f"Não foi possível criar {self.embeddings_file}")

        self._load_embeddings()

    def _load_embeddings(self):
        """Carrega o dicionário de embeddings em memória."""
        try:
            with open(self.embeddings_file, "rb") as f:
                embeddings_dict = pickle.load(f)
            self.doc_names = list(embeddings_dict.keys())
            self.doc_embeddings = np.array(list(embeddings_dict.values()))
            logging.info(f"{len(self.doc_names)} embeddings carregados de {self.embeddings_file}")
        except Exception as e:
            logging.error(f"Erro ao carregar embeddings: {e}")
            self.doc_names = None
            self.doc_embeddings = None

    def _reduce_dimensionality(self, embeddings, method: str):
        logging.info(f"Reduzindo dimensionalidade com {method.upper()}...")
        try:
            if method == "tsne":
                if embeddings.shape[1] > 50:
                    pca_dim = min(50, embeddings.shape[1])
                    embeddings = PCA(n_components=pca_dim, random_state=42)\
                                 .fit_transform(embeddings)
                perplexity = min(30, max(5, len(embeddings)-1))
                return TSNE(n_components=2, perplexity=perplexity,
                            random_state=42, n_jobs=-1)\
                       .fit_transform(embeddings)

            if method == "umap":
                n_neighbors = min(15, max(5, len(embeddings)-1))
                return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                 random_state=42).fit_transform(embeddings)

            if method == "pca":
                return PCA(n_components=2, random_state=42)\
                       .fit_transform(embeddings)

            logging.error(f"Método de redução desconhecido: {method}")
            return None

        except Exception as e:
            logging.error(f"Erro na redução ({method}): {e}")
            return None

    def _cluster_embeddings(self, embeddings, n_clusters: int):
        logging.info(f"Agrupando em {n_clusters} clusters com K-Means...")
        try:
            labels = KMeans(n_clusters=n_clusters, random_state=42,
                            n_init=10).fit_predict(embeddings)
            return labels
        except Exception as e:
            logging.error(f"Erro no K-Means: {e}")
            return None

    def _extract_cluster_keywords(self, labels, n_clusters):
        stop_words = { ... }  # mesma lista de stopwords de antes

        keywords = {}
        for cid in range(n_clusters):
            docs = [self.doc_names[i] for i, l in enumerate(labels) if l == cid]
            if not docs:
                keywords[cid] = f"Cluster {cid+1} vazio"
                continue

            words = []
            for d in docs:
                tokens = d.replace('.txt','').replace('_',' ').split()
                words += tokens

            freq = {}
            for w in words:
                wl = w.lower()
                if wl not in stop_words and len(wl)>2 and not wl.isdigit():
                    freq[wl] = freq.get(wl,0)+1

            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
            if top:
                keywords[cid] = ", ".join(w for w,_ in top)
            else:
                keywords[cid] = docs[0][:15] + "..."
        logging.info("Palavras-chave extraídas.")
        return keywords

    def plot_clean(self, method: str):
        """Scatter simples dos embeddings reduzidos."""
        if self.doc_embeddings is None:
            logging.error("Embeddings não carregados.")
            return None

        red = self._reduce_dimensionality(self.doc_embeddings, method)
        if red is None: return None

        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(red[:,0], red[:,1], alpha=0.7)
        ax.set_title(f"Embeddings ({method.upper()})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        plt.tight_layout()
        return fig

    def plot_clustered(self, method: str, n_clusters: int):
        """Scatter com clusters e convex hulls, e anotações de keywords."""
        if self.doc_embeddings is None:
            logging.error("Embeddings não carregados.")
            return None

        labels = self._cluster_embeddings(self.doc_embeddings, n_clusters)
        if labels is None: return None

        red = self._reduce_dimensionality(self.doc_embeddings, method)
        if red is None: return None

        keywords = self._extract_cluster_keywords(labels, n_clusters)

        fig, ax = plt.subplots(figsize=(18,14))
        colors = plt.cm.tab20(np.linspace(0,1,n_clusters))
        centroids = []

        for i in range(n_clusters):
            pts = red[labels==i]
            if len(pts)>=1:
                centroids.append(pts.mean(axis=0))
                if len(pts)>=3:
                    try:
                        hull = ConvexHull(pts)
                        for s in hull.simplices:
                            ax.fill(pts[s,0], pts[s,1],
                                    color=colors[i], alpha=0.25,
                                    edgecolor='grey', linewidth=0.5)
                    except Exception:
                        pass
            else:
                centroids.append(None)

        ax.scatter(red[:,0], red[:,1],
                   c=[colors[l] for l in labels],
                   s=60, alpha=0.8,
                   edgecolors='black', linewidths=0.5)

        for i, cent in enumerate(centroids):
            if cent is not None:
                ax.annotate(
                    keywords[i], (cent[0], cent[1]),
                    ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5",
                              fc="white", ec=colors[i], lw=2, alpha=0.9)
                )

        ax.set_title(f"Clusters ({method.upper()}, k={n_clusters})", fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        return fig

    def run_interactive_visualization(self):
        """Menu interativo no console para gerar plots."""
        if self.doc_embeddings is None:
            print("⚠ Não foi possível carregar embeddings.")
            return

        print("\n1. Visualização simples\n2. Visualização com clusters")
        choice = input("Opção (1/2): ").strip()
        while choice not in {"1","2"}:
            choice = input("Digite 1 ou 2: ").strip()

        method = input("Método (tsne, umap, pca): ").strip().lower()
        while method not in {"tsne","umap","pca"}:
            method = input("Método inválido. Escolha tsne, umap ou pca: ").strip().lower()

        fig = None
        if choice=="1":
            fig = self.plot_clean(method)
        else:
            k = int(input(f"Número de clusters (2–{len(self.doc_names)}): ").strip())
            fig = self.plot_clustered(method, k)

        if fig:
            print("Mostrando gráfico. Feche a janela para continuar.")
            plt.show()
        else:
            print("Erro ao gerar gráfico.")
