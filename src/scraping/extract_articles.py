import os
import json
import logging
from crossref.restful import Works

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleLinkExtractor:
    """
    Extrai metadados de artigos científicos via CrossRef API e salva URLs de PDF/HTML.
    """

    def __init__(self, query: str, output_file: str = "data/raw/article_links.json", max_results: int = 100):
        self.query = query
        self.output_file = output_file
        self.max_results = max_results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def run(self):
        works = Works()  # instancia a API CrossRef :contentReference[oaicite:1]{index=1}
        logging.info(f"Consultando CrossRef para: '{self.query}' (até {self.max_results} itens)...")
        items = works.query(bibliographic=self.query) \
             .filter(has_full_text="true") \
             .sort('published') \
             .order('desc') \
             .sample(self.max_results)
        links = []
        for item in items:
            title = item.get('title', [""])[0]
            doi   = item.get('DOI')
            # tenta obter URL direto (p.ex. DOI link ou URL no JSON)
            url   = item.get('URL') or f"https://doi.org/{doi}"
            links.append({"title": title, "doi": doi, "url": url})
        # salva JSON
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(links, f, indent=4, ensure_ascii=False)
        logging.info(f"{len(links)} links de artigos salvos em {self.output_file}")
