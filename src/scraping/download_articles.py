import os
import json
import logging
from typing import Dict, Any

import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# Configure logging\logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ArticleDownloader:
    """
    Downloads scientific articles from URLs and extracts their text.
    Supports PDF via PyPDF2 and HTML via BeautifulSoup.
    """

    def __init__(
        self,
        links_file: str = "data/raw/article_links.json",
        output_folder: str = "data/raw/articles/"
    ):
        self.links_file = links_file
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and strip leading/trailing spaces."""
        return " ".join(text.split()).strip()

    def extract_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file using PyPDF2."""
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def download_and_process(
        self,
        record: Dict[str, Any],
        index: int,
        total: int
    ) -> None:
        """
        Downloads a single record and saves its extracted text as .txt.
        Handles both PDF and HTML content.
        """
        title = record.get("title", f"article_{index}")
        safe_title = title.replace(" ", "_").lower()
        url = record.get("url")
        logging.info(f"[{index+1}/{total}] Processing '{title}'")

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.output_folder, f"{safe_title}.pdf")
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(response.content)
                raw_text = self.extract_pdf(pdf_path)
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                paragraphs = soup.find_all("p")
                raw_text = " ".join(p.get_text() for p in paragraphs)

            text = self.clean_text(raw_text)
            if not text:
                logging.warning(f"[{index+1}/{total}] No extractable text for '{title}'")
                return

            txt_path = os.path.join(self.output_folder, f"{safe_title}.txt")
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
            logging.info(f"[{index+1}/{total}] Saved text to '{txt_path}'")

        except Exception as e:
            logging.error(f"[{index+1}/{total}] Failed to process '{title}': {e}")

    def run(self) -> None:
        """Reads link records and processes each article."""
        if not os.path.exists(self.links_file):
            logging.error(f"Links file not found: {self.links_file}")
            return

        with open(self.links_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        total = len(records)
        logging.info(f"Starting download of {total} articles.")
        for idx, record in enumerate(records):
            self.download_and_process(record, idx, total)

        logging.info("Article download and extraction complete.")

if __name__ == "__main__":
    downloader = ArticleDownloader()
    downloader.run()
