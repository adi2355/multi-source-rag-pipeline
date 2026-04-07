# ArXiv Paper Collector

This utility allows you to download, process, and analyze research papers from ArXiv using the MistralOCR system.

## Features

- Download papers from ArXiv based on categories or search queries
- Download specific papers by URL
- Process papers using Mistral OCR for high-quality text extraction
- Store paper metadata and full text in a SQLite database
- Avoid duplicates by checking against existing database entries

## Installation

Make sure you have installed all the required dependencies:

```bash
pip install arxiv requests PyPDF2 mistralai
```

## Usage

The `arxiv_collector.py` script provides several command-line options:

### Download Papers

Download papers from ArXiv based on categories defined in `config.py`:

```bash
python arxiv_collector.py --download-only --max 5
```

Download papers with a custom search query:

```bash
python arxiv_collector.py --download-only --max 5 --search-query "selective state space AND ti:mamba"
```

### Download Papers by URL

Download a single paper by URL:

```bash
python arxiv_collector.py --download-url https://arxiv.org/abs/2312.11805
```

Download multiple papers from a file containing URLs (one per line):

```bash
python arxiv_collector.py --download-urls-file paper_urls.txt
```

Example `paper_urls.txt` file:
```
https://arxiv.org/abs/2406.00003
https://arxiv.org/abs/2312.11805
https://arxiv.org/abs/2403.00815
```

### Process Downloaded Papers

Process papers that have been downloaded but not yet inserted into the database:

```bash
python run.py --process-papers
```

This command will:
1. Extract text from PDFs using Mistral OCR
2. Store the metadata and text in the database
3. Update the pending papers list

## How it Works

1. **Paper Download:**
   - Papers are downloaded from ArXiv using either search queries or direct URLs
   - Each paper's PDF is saved to the `data/papers/pdf/` directory
   - Metadata is stored in `data/papers/pending_papers.json`

2. **Duplicate Checking:**
   - The system checks for duplicates by ArXiv ID in both the pending list and the database
   - Duplicate PDFs are skipped and optionally deleted to save space

3. **Text Extraction:**
   - Mistral OCR is used for high-quality text extraction from PDFs
   - If Mistral OCR fails, the system falls back to PyPDF2

4. **Database Storage:**
   - Paper metadata and extracted text are stored in the `research_papers` table
   - ArXiv IDs are stored in the `doi` field to maintain uniqueness

## Configuration

Key configuration options in `config.py`:

```python
# ArXiv settings
ARXIV_CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE']
ARXIV_CONFIG = {
    'search_query': '"selective state space" AND ti:mamba',
    'max_results': 100
}

# Mistral OCR settings
MISTRAL_API_KEY = 'your-api-key-here'
MISTRAL_MODEL = 'mistral-ocr-latest'
```

## Troubleshooting

- **No papers downloaded:** Check your search query or ArXiv categories in config.py
- **OCR failures:** Ensure your Mistral API key is correctly set in config.py
- **Database errors:** Check the database schema with `sqlite3 data/knowledge_base.db '.schema research_papers'` 