import os
import json
import logging
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleDownloader:
    """
    Baixa PDFs de artigos e extrai texto. Usa PyPDF2 para PDFs e BeautifulSoup para HTML.
    """

    def __init__(self, links_file="data/raw/article_links.json", output_folder="data/raw/articles/"):
        self.links_file = links_file
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _extract_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)           # lê PDF com PyPDF2 :contentReference[oaicite:5]{index=5}
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)

    def _download_and_process(self, record: dict, index: int, total: int):
        title = record.get("title", f"article_{index}")
        safe_title = title.replace(" ", "_").lower()
        url = record["url"]
        logging.info(f"[{index+1}/{total}] Processando: {title}")

        try:
            resp = requests.get(url, timeout=20, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            # 1) Se for PDF
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.output_folder, f"{safe_title}.pdf")
                with open(pdf_path, "wb") as f_pdf:
                    f_pdf.write(resp.content)
                text = self._extract_pdf(pdf_path)

            # 2) Caso contrário, trata como página HTML (extraindo <p>)
            else:
                soup = BeautifulSoup(resp.content, "html.parser")
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text() for p in paragraphs)

            cleaned = self._clean_text(text)
            if not cleaned:
                logging.warning(f"[{index+1}/{total}] Sem texto extraído para {title}")
                return

            txt_path = os.path.join(self.output_folder, f"{safe_title}.txt")
            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write(cleaned)
            logging.info(f"[{index+1}/{total}] Texto salvo em {txt_path}")

        except Exception as e:
            logging.error(f"[{index+1}/{total}] Erro em {title}: {e}")

    def run(self):
        if not os.path.exists(self.links_file):
            logging.error("Arquivo de links não encontrado.")
            return
        with open(self.links_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        total = len(records)
        for i, rec in enumerate(records):
            self._download_and_process(rec, i, total)
        logging.info("Processamento concluído.")
