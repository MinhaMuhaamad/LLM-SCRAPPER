import asyncio
import os
import re
import aiofiles
import aiohttp
import csv
import queue
import threading
from datetime import datetime
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import google.generativeai as genai
class NeurIPSAsyncScraper:
    BASE_URL = "https://papers.nips.cc"
    MAX_RETRIES = 5
    TIMEOUT = 180
    CONCURRENT_LIMIT = 2
    RETRY_BACKOFF = [2, 4, 8, 16, 32]
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    ANNOTATION_CATEGORIES = ["Deep Learning", "Reinforcement Learning", "NLP", "Computer Vision", "Optimization"]
    def __init__(self, gemini_api_key):
        self.base_pdfs_dir = "neurips_papers"
        os.makedirs(self.base_pdfs_dir, exist_ok=True)
        self.metadata_queue = queue.Queue()
        self.stop_metadata_writer = threading.Event()
        self.connector = aiohttp.TCPConnector(limit=self.CONCURRENT_LIMIT)
        self.failed_dir = os.path.join(self.base_pdfs_dir, "failed_metadata")
        os.makedirs(self.failed_dir, exist_ok=True)
        # Initialize Google Gemini API
        genai.configure(api_key='AIzaSyCD1168rI5Hf0sukzhcXxYmdUOzFzfDBhU')
    def metadata_writer(self):
        csv_path = os.path.join(self.base_pdfs_dir, "annotated_papers.csv")
        file_exists = os.path.isfile(csv_path)
        
        while not self.stop_metadata_writer.is_set() or not self.metadata_queue.empty():
            try:
                metadata = self.metadata_queue.get(timeout=1)
                with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['Year', 'Title', 'PDF Link', 'Authors', 'Download Time', 'Annotation'])
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True
                    writer.writerow(metadata)
                print(f"Metadata saved: {metadata['Title']}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in metadata writer: {e}")

    async def scrape_neurips_papers(self):
        current_year = 2024
        start_year = current_year - 5
        metadata_thread = threading.Thread(target=self.metadata_writer)
        metadata_thread.start()
        timeout = aiohttp.ClientTimeout(total=self.TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=self.connector, headers=self.HEADERS) as session:
                tasks = [self.process_year(session, year) for year in range(start_year, current_year + 1)]
                await asyncio.gather(*tasks)
        finally:
            self.stop_metadata_writer.set()
            metadata_thread.join()

    async def process_year(self, session, year):
        url = f"{self.BASE_URL}/paper_files/paper/{year}"
        year_folder = os.path.join(self.base_pdfs_dir, str(year))
        os.makedirs(year_folder, exist_ok=True)

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to fetch {url}, Status: {response.status}")
                    return
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                papers = soup.select('body > div.container-fluid > div > ul  li a[href]')
                if not papers:
                    print(f"No papers found for {year}.")
                    return
                
                print(f"Found {len(papers)} papers for {year}")
                tasks = [self.process_paper(session, year, paper, year_folder) for paper in papers]
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error processing {year}: {e}")

    async def process_paper(self, session, year, paper, year_folder):
        title = paper.text.strip()
        paper_url = f"{self.BASE_URL}{paper['href']}"
        pdf_url = paper_url.replace('hash/', 'file/').replace("Abstract", "Paper").replace('.html', '.pdf')
        authors = await self.extract_authors(session, paper_url)

        filename = re.sub(r'[<>:"/\\|?*]', '_', title) + ".pdf"
        file_path = os.path.join(year_folder, filename)

        if await self.download_pdf(session, pdf_url, file_path):
            # Extract authors from the PDF if not found in HTML
            if authors == "Unknown Authors":
                authors = self.extract_authors_from_pdf(file_path)
            
            annotation = self.annotate_paper(file_path)
            metadata = {
                'Year': year,
                'Title': title,
                'PDF Link': pdf_url,
                'Authors': authors,
                'Download Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Annotation': annotation
            }
            self.metadata_queue.put(metadata)

    async def extract_authors(self, session, paper_url):
        try:
            async with session.get(paper_url) as response:
                if response.status != 200:
                    print(f"Failed to fetch {paper_url}, Status: {response.status}")
                    return "Unknown Authors"
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract authors using the provided selector
                authors = ", ".join([a.text.strip() for a in soup.select(".authors a")]) or "Unknown Authors"
                return authors
        except Exception as e:
            print(f"Error extracting authors: {e}")
            return "Unknown Authors"

    def extract_authors_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            # Look for patterns indicating author names
            author_patterns = [
                r"Authors?:\s*([\w\s,]+)",  # Matches "Authors: John Doe, Jane Smith"
                r"By:\s*([\w\s,]+)",        # Matches "By: John Doe, Jane Smith"
                r"[\w\s,]+\([\w\s,]+\)"     # Matches "John Doe (University), Jane Smith (Company)"
            ]

            for pattern in author_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return "Unknown Authors"
        except Exception as e:
            print(f"Error extracting authors from PDF: {e}")
            return "Unknown Authors"

    async def download_pdf(self, session, pdf_url, file_path):
        try:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, mode='wb') as f:
                        await f.write(await response.read())
                    print(f"Downloaded: {file_path}")
                    return True
                else:
                    print(f"Failed to download {file_path}. Status: {response.status}")
                    return False
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def annotate_paper(self, file_path):
        try:
            text = self.extract_text_from_pdf(file_path)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(f"Classify this research paper into one of the following categories: {self.ANNOTATION_CATEGORIES}.\n\n{text}")
            return response.text.strip()
        except Exception as e:
            print(f"Annotation failed: {e}")
            return "Unknown"

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text[:3000]  # Limit input text size
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""
async def main():
    scraper = NeurIPSAsyncScraper(gemini_api_key="YOUR_GEMINI_API_KEY")
    await scraper.scrape_neurips_papers()
if __name__ == "__main__":
    asyncio.run(main())
