import os
import logging
import re
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TextCleaner:
    """
    Responsible for cleaning raw text files in a specified directory.
    """

    def __init__(
        self,
        raw_docs_folder: str = "data/raw/articles/",
        processed_docs_folder: str = "data/processed/"
    ):
        """
        Initializes the cleaner.

        Args:
            raw_docs_folder: Directory containing unprocessed text files.
            processed_docs_folder: Directory where cleaned files will be saved.
        """
        self.raw_docs_folder = raw_docs_folder
        self.processed_docs_folder = processed_docs_folder
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Ensures the processed_docs_folder exists."""
        try:
            os.makedirs(self.processed_docs_folder, exist_ok=True)
            logging.info(f"Verified or created output folder: {self.processed_docs_folder}")
        except Exception as e:
            logging.error(f"Failed to create output folder {self.processed_docs_folder}: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans the input text by removing unwanted patterns and normalizing whitespace.

        The function removes navigation menus, multiple whitespace runs, and non-ASCII characters.

        Args:
            text: The raw text string.

        Returns:
            Cleaned text string.
        """
        # Remove GitHub navigation menus
        text = re.sub(
            r"Navigation Menu.*?(Explore All features|View source on GitHub)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        return text.strip()

    def _list_raw_files(self) -> List[str]:
        """Lists all files in the raw_docs_folder."""
        if not os.path.isdir(self.raw_docs_folder):
            logging.error(f"Raw docs folder not found: {self.raw_docs_folder}")
            return []
        files = []
        try:
            for filename in os.listdir(self.raw_docs_folder):
                full_path = os.path.join(self.raw_docs_folder, filename)
                if os.path.isfile(full_path):
                    files.append(filename)
            logging.info(f"Found {len(files)} files in {self.raw_docs_folder}")
        except Exception as e:
            logging.error(f"Error listing files in {self.raw_docs_folder}: {e}")
        return files

    def process_file(self, filename: str) -> bool:
        """
        Reads, cleans, and writes one file.

        Args:
            filename: Name of the file to process.

        Returns:
            True if successful, False if any error occurred or result is empty.
        """
        src = os.path.join(self.raw_docs_folder, filename)
        dst = os.path.join(self.processed_docs_folder, filename)
        try:
            with open(src, "r", encoding="utf-8") as f:
                raw = f.read()
            cleaned = self.clean_text(raw)
            if not cleaned:
                logging.warning(f"{filename} is empty after cleaning; skipping.")
                return False
            with open(dst, "w", encoding="utf-8") as f:
                f.write(cleaned)
            return True
        except FileNotFoundError:
            logging.error(f"Source file not found: {src}")
            return False
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")
            return False

    def run(self) -> None:
        """
        Processes all files in the raw_docs_folder and writes cleaned versions.
        """
        files = self._list_raw_files()
        if not files:
            logging.warning(f"No files to process in {self.raw_docs_folder}.")
            return
        logging.info(f"Starting cleaning of {len(files)} files...")
        success = 0
        failures = 0
        for i, fname in enumerate(files, 1):
            if self.process_file(fname):
                logging.info(f"[{i}/{len(files)}] Cleaned: {fname}")
                success += 1
            else:
                logging.error(f"[{i}/{len(files)}] Failed: {fname}")
                failures += 1
        logging.info(f"Cleaning complete. {success} succeeded, {failures} failed.")

if __name__ == "__main__":
    cleaner = TextCleaner()
    cleaner.run()
