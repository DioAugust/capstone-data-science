import os
import pickle
import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SemanticSearcher:
    """
    Handles loading document embeddings, performing semantic search
    and returning the top matching documents.
    """

    def __init__(
        self,
        embeddings_file: str = "embeddings/document_embeddings.pkl",
        processed_docs_folder: str = "data/processed/",
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initializes the SemanticSearcher.

        Args:
            embeddings_file: Path to the pickle file with document embeddings.
            processed_docs_folder: Directory where cleaned articles are stored.
            model_name: SentenceTransformer model identifier (must match generator).
        """
        self.embeddings_file = embeddings_file
        self.processed_docs_folder = processed_docs_folder
        self.model_name = model_name
        self.model = self._load_model()
        self.doc_names: List[str] = []
        self.doc_embeddings: np.ndarray = np.array([])
        self._load_embeddings()

    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model for queries."""
        logging.info(f"Loading search model: {self.model_name}")
        try:
            model = SentenceTransformer(self.model_name)
            logging.info("Search model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}': {e}")
            raise

    def _load_embeddings(self) -> None:
        """Loads precomputed embeddings from disk."""
        if not os.path.exists(self.embeddings_file):
            logging.error(
                f"Embeddings file not found: {self.embeddings_file}."
                " Generate embeddings before searching."
            )
            return
        try:
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
            self.doc_names = list(data.keys())
            self.doc_embeddings = np.array(list(data.values()))
            # Ensure document embeddings are normalized as well
            self.doc_embeddings = normalize(self.doc_embeddings)
            logging.info(f"Loaded and normalized {len(self.doc_names)} embeddings.")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")

    def search(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Performs a semantic search against loaded document embeddings.

        Args:
            query: Natural language search string.
            top_n: Number of top results to return.

        Returns:
            A list of (document_name, similarity_score).
        """
        if not self.doc_names or self.doc_embeddings.size == 0:
            logging.error("Embeddings not loaded properly; cannot search.")
            return []
        if not query.strip():
            logging.warning("Empty query provided; no results.")
            return []

        logging.info(f"Running semantic search for: '{query}'")
        try:
            # 1) Compute and normalize query embedding
            q_emb = self.model.encode(query, convert_to_numpy=True)
            q_emb = normalize(q_emb.reshape(1, -1))[0]
            # 2) Cosine similarity against normalized docs
            sims = cosine_similarity([q_emb], self.doc_embeddings)[0]
            # 3) Limit top_n to available docs
            n = min(top_n, len(self.doc_names))
            # 4) Get highest scoring indices
            idx = np.argsort(sims)[-n:][::-1]
            results = [(self.doc_names[i], float(sims[i])) for i in idx]
            logging.info(f"Search complete: returned {len(results)} results.")
            return results
        except Exception as e:
            logging.error(f"Search failed for query '{query}': {e}")
            return []

    def display_document(self, doc_name: str, max_chars: int = 1000) -> None:
        """
        Prints the start of the specified document to console.

        Args:
            doc_name: Filename of the document to show.
            max_chars: Maximum characters to print.
        """
        if doc_name not in self.doc_names:
            logging.error(f"Document '{doc_name}' not in loaded embeddings.")
            return
        path = os.path.join(self.processed_docs_folder, doc_name)
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"\nDocument: {doc_name}\n")
            print(text[:max_chars])
            if len(text) > max_chars:
                print("\n...content truncated...")
        except Exception as e:
            logging.error(f"Error reading '{doc_name}': {e}")

    def interactive(self) -> None:
        """
        Runs a simple CLI loop for interactive semantic searching.
        """
        if not self.doc_names:
            logging.critical("No embeddings for interactive search.")
            return
        print("Semantic Search CLI (type 'exit' to quit)")
        while True:
            query = input("Search> ").strip()
            if query.lower() == 'exit':
                break
            results = self.search(query)
            if not results:
                print("No relevant documents found.")
                continue
            for i, (name, score) in enumerate(results, 1):
                print(f"{i}. {name} ({score:.4f})")
            choice = input("Enter number to view or 'back': ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    self.display_document(results[idx][0], max_chars)

if __name__ == '__main__':
    searcher = SemanticSearcher()
    searcher.interactive()
