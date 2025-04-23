import os
import time
import subprocess

from src.scraping.extract_articles import ArticleLinkExtractor
from src.scraping.download_articles import ArticleDownloader
from src.processing.text_cleaning import TextCleaner
from src.processing.generate_embeddings import EmbeddingGenerator

# Main file paths
LINKS_FILE = "data/raw/article_links.json"
RAW_ARTICLES_FOLDER = "data/raw/articles/"
PROCESSED_ARTICLES_FOLDER = "data/processed/"
EMBEDDINGS_FILE = "embeddings/document_embeddings.pkl"


def run_pipeline():
    """Runs the full pipeline before launching the Streamlit dashboard."""
    print("\n🚀 Starting scientific articles processing pipeline...\n")

    # 1. Extract article links via CrossRef if not already present
    if not os.path.exists(LINKS_FILE):
        print("🔗 Extracting article links...")
        try:
            extractor = ArticleLinkExtractor(
                query="machine learning",
                output_file=LINKS_FILE,
                max_results=100
            )
            extractor.run()
            if os.path.exists(LINKS_FILE):
                print(f"✅ Links extracted and saved to {LINKS_FILE}")
            else:
                print("⚠ Extraction completed but links file is missing.")
                return
        except Exception as err:
            print(f"❌ Error extracting links: {err}")
            return
    else:
        print("✅ Article links already exist. Skipping link extraction.")

    time.sleep(1)

    # 2. Download and extract text from articles (PDF/HTML)
    if not os.path.exists(RAW_ARTICLES_FOLDER) or not os.listdir(RAW_ARTICLES_FOLDER):
        print("\n📥 Downloading and processing articles...")
        try:
            downloader = ArticleDownloader(
                links_file=LINKS_FILE,
                output_folder=RAW_ARTICLES_FOLDER
            )
            downloader.run()
            if os.path.exists(RAW_ARTICLES_FOLDER) and os.listdir(RAW_ARTICLES_FOLDER):
                print("✅ Download and extraction complete.")
            else:
                print("⚠ Download finished but raw articles folder is empty.")
        except Exception as err:
            print(f"❌ Error downloading or extracting articles: {err}")
    else:
        print("✅ Articles already downloaded. Skipping this step.")

    time.sleep(1)

    # 3. Clean and structure the article texts
    if not os.path.exists(PROCESSED_ARTICLES_FOLDER) or not os.listdir(PROCESSED_ARTICLES_FOLDER):
        print("\n🧹 Cleaning and structuring article texts...")
        try:
            cleaner = TextCleaner(
                raw_docs_folder=RAW_ARTICLES_FOLDER,
                processed_docs_folder=PROCESSED_ARTICLES_FOLDER
            )
            cleaner.run()
            if os.path.exists(PROCESSED_ARTICLES_FOLDER) and os.listdir(PROCESSED_ARTICLES_FOLDER):
                print("✅ Text cleaning complete.")
            else:
                print("⚠ Cleaning finished but processed folder is empty.")
        except Exception as err:
            print(f"❌ Error cleaning texts: {err}")
    else:
        print("✅ Texts already cleaned. Skipping this step.")

    time.sleep(1)

    # 4. Generate embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print("\n🧠 Generating document embeddings...")
        try:
            generator = EmbeddingGenerator(
                processed_docs_folder=PROCESSED_ARTICLES_FOLDER,
                embeddings_file=EMBEDDINGS_FILE
            )
            generator.run()
            if os.path.exists(EMBEDDINGS_FILE):
                print("✅ Embeddings generation complete.")
            else:
                print("⚠ Embeddings generated but file not found.")
                return
        except Exception as err:
            print(f"❌ Error generating embeddings: {err}")
            return
    else:
        print("✅ Embeddings already exist. Skipping this step.")

    print("\n✅ Pipeline finished successfully!\n")


def start_dashboard():
    """Launches the Streamlit dashboard."""
    print("\n🌐 Starting Streamlit dashboard...\n")
    subprocess.run([
        "streamlit", "run", "app/dashboard.py",
        "--server.fileWatcherType", "none"
    ])


if __name__ == "__main__":
    run_pipeline()
    start_dashboard()
