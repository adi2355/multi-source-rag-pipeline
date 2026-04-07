# Instagram Scraper

Instagram scraper that works around rate limiting. Download, transcribe, and analyze Instagram content using AI.

## Features

- **Instagram Content Downloading**: Download videos and metadata from Instagram accounts
- **Audio Transcription**: Extract audio and transcribe using Whisper
- **AI-Powered Summarization**: Generate structured summaries with Claude
- **Batch Processing**: Cost-efficient processing of transcripts using Claude Batches API
- **Search & Retrieval**: Index and search content with full-text search
- **Research Paper Collection**: Download papers from ArXiv and custom URLs
- **Proxy Support**: Download content through proxies to avoid rate limiting
- **Advanced OCR**: Use Mistral OCR for superior text extraction from PDFs

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Instagram-Scraper.git
   cd Instagram-Scraper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your configuration in `config.py` including:
   - Instagram credentials
   - Anthropic API key for Claude

## Usage

### Basic Commands

```bash
# Run all steps
python run.py --all

# Download content from Instagram
python run.py --download

# Extract and transcribe audio from videos
python run.py --transcribe 

# Generate AI summaries of transcripts
python run.py --summarize

# Run the web interface
python run.py --web
```

### Advanced Options

```bash
# Disable batch processing for summarization (costs more but may be faster for small datasets)
python run.py --summarize --no-batch

# Force refresh Instagram content
python run.py --download --refresh-force

# Run Instagram download without authentication 
python run.py --download --no-auth

# Collect research papers from ArXiv and custom URLs
python run.py --papers
```

## Research Paper Collection

The system can download and process research papers from multiple sources:

### ArXiv Collection

Papers are collected from ArXiv based on configured AI/ML topics in `config.py`.

### Proxy Support for Downloads

All paper downloads now support proxies to help with rate limiting and geo-restricted content:

- Configure proxies in `config.py` under `PROXY_SERVERS` or `PROXY_CONFIG`
- The system will automatically use configured proxies for all HTTP requests
- Proxy rotation support helps avoid rate limiting

### Mistral OCR for Document Understanding

The system now integrates with Mistral OCR for superior text extraction from PDFs:

- **Enhanced Text Extraction**: Improved extraction quality especially for complex layouts
- **Document Understanding**: Better handling of tables, charts, and formatted text
- **Fallback Mechanism**: Automatically falls back to PyPDF2 if OCR fails or API is unavailable

#### Configuration

Edit `config.py` to configure Mistral OCR:

```python
MISTRAL_OCR_CONFIG = {
    'enabled': True,  # Enable/disable this feature
    'model': 'mistral-ocr-latest',
    'fallback_to_pypdf': True,  # If OCR fails, fall back to PyPDF2
    'include_images': False,    # Whether to include base64 images in extraction
    'api_key': os.getenv("MISTRAL_API_KEY", "your_mistral_api_key")
}
```

### Custom Paper URL Support

The system now supports downloading papers from arbitrary URLs:

1. **Direct PDF Links**: Provide direct links to PDF files
2. **Webpage Links**: The system will intelligently extract PDF links from academic pages

#### How It Works

- URLs are configured in the `paper_urls` list in `config.py`
- For webpage URLs, the system scans for PDF download links
- Papers are processed, their text extracted, and added to the knowledge base
- Metadata (title, authors, abstract) is intelligently extracted from the content

#### Configuration

Edit `config.py` to add your paper URLs:

```python
RESEARCH_PAPER_CONFIG = {
    # other settings...
    'enable_custom_paper_urls': True,  # Enable/disable this feature
    'paper_urls': [
        "https://arxiv.org/pdf/2307.09288.pdf",  # Direct PDF link
        "https://proceedings.neurips.cc/paper_files/paper/2023/file/sample-Paper.pdf",
        "https://www.example.com/research-papers/sample-paper",  # Webpage containing PDF
    ]
}
```

## AI Summarization with Claude Batch Processing

The system uses Claude to analyze and summarize video transcripts. By default, it uses Claude's Message Batches API, which offers:

- **50% Cost Reduction**: Batch processing is half the cost of regular API calls
- **Asynchronous Processing**: Process large volumes of transcripts efficiently
- **Structured Output**: Generates structured summaries with key topics, insights and more

### How Batch Processing Works

1. Transcripts are collected and prepared in batches of up to 100 items
2. The batch is submitted to Claude's Message Batches API for asynchronous processing
3. The system polls for batch completion and processes results when ready
4. Summaries are stored in the database and cached for future use

### Cost Efficiency

With batch processing enabled (default), Claude API costs are reduced by 50%:

| Model              | Batch Input   | Batch Output  | Standard Input | Standard Output |
|--------------------|---------------|---------------|----------------|-----------------|
| Claude 3.7 Sonnet  | $1.50 / MTok  | $7.50 / MTok  | $3.00 / MTok   | $15.00 / MTok   |
| Claude 3.5 Sonnet  | $1.50 / MTok  | $7.50 / MTok  | $3.00 / MTok   | $15.00 / MTok   |
| Claude 3 Haiku     | $0.125 / MTok | $0.625 / MTok | $0.25 / MTok   | $1.25 / MTok    |

The system uses Claude 3 Haiku by default for optimal cost efficiency.

## Enhanced Paper Collection

The system now includes advanced paper collection capabilities with the following features:

### Duplicate Detection

The system now intelligently detects duplicate papers using multiple methods:
- **Title Similarity**: Uses Levenshtein distance to detect papers with similar titles
- **Content Similarity**: Compares document content using Jaccard similarity on text chunks
- **Automatic Cleanup**: Automatically removes duplicate PDFs to save storage space

### Custom Paper URLs

You can now add papers from any source, not just ArXiv:
- **Direct PDF Links**: Add direct links to PDF files
- **Webpage Parsing**: Automatically extracts PDF links from webpages
- **Flexible Configuration**: Enable/disable this feature in config.py

To add custom paper URLs, edit the `config.py` file:

```python
# Custom Paper URLs Configuration
ENABLE_PAPER_URLS = True
PAPER_URLS = [
    "https://arxiv.org/pdf/2303.08774.pdf",  # Example: "GPT-4 Technical Report"
    "https://example.com/research-paper",    # Example: Webpage containing a PDF
]
```

### ArXiv Integration

The system includes improved ArXiv integration:
- **Category-Based Collection**: Collect papers by ArXiv category
- **Configurable Results**: Set maximum results per category
- **Flexible Configuration**: Enable/disable ArXiv collection in config.py

Configure ArXiv collection in `config.py`:

```python
# Paper Collection Configuration
ENABLE_ARXIV_COLLECTION = True
ARXIV_CATEGORIES = [
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language
    "cs.CV",  # Computer Vision
]
ARXIV_MAX_RESULTS = 10  # Maximum results per category
```

## Advanced OCR with Mistral

This project includes integration with Mistral OCR for enhanced PDF text extraction, providing:

- **Improved extraction quality** over traditional PDF parsers
- **Support for large PDFs** through automatic chunking
- **Batch processing capabilities** for efficient handling of multiple documents
- **Automatic fallback** to PyPDF2 if Mistral OCR fails

### Batch Processing Papers with Mistral OCR

The system now supports a two-stage process for efficient paper processing:

1. **Download-only mode**: Download papers without processing them
   ```bash
   python run.py --download-papers --max-papers 50
   ```

2. **Batch processing mode**: Process previously downloaded papers in batch
   ```bash
   python run.py --process-papers --max-papers 20
   ```

This separation allows you to:
- Download papers during off-peak hours or when you have good internet
- Process with Mistral OCR when you have API quota available
- Resume processing if it was interrupted
- Handle large collections of papers more efficiently

### Configuration

Configure Mistral OCR in `config.py`:

```python
# Enable/disable Mistral OCR
ENABLE_MISTRAL_OCR = True

# Your Mistral API key
MISTRAL_API_KEY = "your-api-key-here"

# Model selection
MISTRAL_MODEL = "mistral-large-latest"

# Processing parameters
MISTRAL_CHUNK_SIZE = 15000
MISTRAL_MAX_RETRIES = 3
MISTRAL_RETRY_DELAY = 2

# Enable fallback to PyPDF2 if Mistral OCR fails
MISTRAL_FALLBACK_TO_PYPDF = True

# Batch processing settings
BATCH_SIZE = 5
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
