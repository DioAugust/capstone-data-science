import os
import pickle
import logging
from typing import Dict

import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Configure logging
tlogging = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class EmbeddingGenerator:
    """
    Reads processed text files, generates sentence embeddings,
    normalizes them, and saves to a pickle file.
    """
    def __init__(
        self,
        processed_docs_folder: str = "data/processed/",
        embeddings_file: str = "embeddings/document_embeddings.pkl",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32
    ):
        """
        Initializes the embedding generator.

        Args:
            processed_docs_folder: Directory containing cleaned text files.
            embeddings_file: Path where embeddings pickle will be saved.
            model_name: SentenceTransformer model identifier (must match searcher).
            batch_size: Number of documents to batch during encoding.
        """
        self.processed_docs_folder = processed_docs_folder
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        self.batch_size = batch_size
        self.embeddings_dir = os.path.dirname(embeddings_file)
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model."""
        logging.info(f"Loading embedding model: {self.model_name}")
        try:
            model = SentenceTransformer(self.model_name)
            logging.info("Embedding model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}': {e}")
            raise

    def _read_documents(self) -> Dict[str, str]:
        """Reads all text files from the processed_docs_folder."""
        docs = {}
        if not os.path.isdir(self.processed_docs_folder):
            logging.error(f"Processed docs folder not found: {self.processed_docs_folder}")
            return docs
        for fname in sorted(os.listdir(self.processed_docs_folder)):
            path = os.path.join(self.processed_docs_folder, fname)
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        docs[fname] = f.read()
                except Exception as e:
                    logging.warning(f"Could not read '{fname}': {e}")
        logging.info(f"Read {len(docs)} documents.")
        return docs

    def _generate_embeddings(self, docs: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Generates and normalizes embeddings for each document."""
        if not docs:
            logging.warning("No documents provided to generate embeddings.")
            return {}
        texts = list(docs.values())
        names = list(docs.keys())
        logging.info("Generating embeddings in batches...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        # Normalize vectors to unit length for cosine similarity
        embeddings = normalize(embeddings)
        logging.info(f"Generated embeddings shape: {embeddings.shape}")
        return dict(zip(names, embeddings))

    def _save_embeddings(self, embedding_dict: Dict[str, np.ndarray]) -> None:
        """Saves the embedding dictionary to a pickle file."""
        if not embedding_dict:
            logging.warning("No embeddings to save.")
            return
        os.makedirs(self.embeddings_dir, exist_ok=True)
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embedding_dict, f)
            logging.info(f"Embeddings saved to {self.embeddings_file}")
        except Exception as e:
            logging.error(f"Failed to save embeddings: {e}")

    def run(self) -> None:
        """
        Executes the full pipeline: read docs, generate embeddings, and save.
        """
        docs = self._read_documents()
        if not docs:
            logging.error("Embedding generation aborted: no documents found.")
            return
        embed_dict = self._generate_embeddings(docs)
        if not embed_dict:
            logging.error("Embedding generation aborted: generation failed.")
            return
        self._save_embeddings(embed_dict)
        logging.info("Embedding pipeline completed.")

if __name__ == '__main__':
    gen = EmbeddingGenerator()
    gen.run()
