import os
import json
import logging
from typing import List, Dict, Any
from crossref.restful import Works

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleLinkExtractor:
    """
    Extracts metadata for scientific articles using the CrossRef API
    and saves their PDF/HTML URLs to a JSON file.
    """

    def __init__(
        self,
        query: str,
        output_file: str = "data/raw/article_links.json",
        max_results: int = 100
    ):
        self.query = query
        self.output_file = output_file
        self.max_results = max_results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def run(self) -> None:
        """
        Executes the query against CrossRef, filters results for full text,
        and writes a list of dicts containing title, DOI, and URL to JSON.
        """
        works = Works()
        logging.info(f"Querying CrossRef for '{self.query}' (up to {self.max_results} items)...")
        items = (
            works.query(bibliographic=self.query)
            .filter(has_full_text="true")
            .sort('published')
            .order('desc')
            .sample(self.max_results)
        )

        links: List[Dict[str, Any]] = []
        for item in items:
            title_list = item.get('title', [])
            title = title_list[0] if title_list else 'Unknown Title'
            doi = item.get('DOI', '')
            url = item.get('URL', '') or f"https://doi.org/{doi}" if doi else ''
            if not url:
                logging.warning(f"Skipping entry with missing URL or DOI: {item}")
                continue
            links.append({
                'title': title,
                'doi': doi,
                'url': url
            })

        # Save to JSON
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(links, f, indent=4, ensure_ascii=False)
            logging.info(f"Successfully saved {len(links)} article links to {self.output_file}")
        except Exception as e:
            logging.error(f"Failed to write links to {self.output_file}: {e}")

if __name__ == '__main__':
    extractor = ArticleLinkExtractor(query='machine learning')
    extractor.run()
