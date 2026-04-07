#!/usr/bin/env python3
"""
Script to compare PDF files and text files, adding PDFs without corresponding text files
to the pending list for processing.
"""
import os
import json
import logging
import re
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('add_missing_text_files')

def normalize_filename(filename):
    """
    Normalize filename for comparison by removing extension and 
    standardizing ArXiv IDs and hash formats.
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # ArXiv ID pattern (e.g., 1234.56789v1)
    if re.match(r'^\d+\.\d+v\d+$', base_name):
        return base_name
    
    # MD5 hash pattern (32 hex characters)
    if re.match(r'^[0-9a-f]{32}$', base_name):
        return base_name
    
    # Other pattern (just return as is)
    return base_name

def main():
    """Main function to compare PDF and text files and update the pending list."""
    parser = argparse.ArgumentParser(description='Add PDFs without text files to the pending list')
    parser.add_argument('--pdf-dir', type=str, 
                        default='/home/adi235/MistralOCR/Instagram-Scraper/data/papers/pdf',
                        help='Directory containing PDF files')
    parser.add_argument('--text-dir', type=str, 
                        default='/home/adi235/MistralOCR/Instagram-Scraper/data/papers/text',
                        help='Directory containing text files')
    parser.add_argument('--pending-file', type=str, 
                        default='/home/adi235/MistralOCR/Instagram-Scraper/data/papers/pending_papers.json',
                        help='Path to the pending_papers.json file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--max-papers', type=int, default=None,
                        help='Maximum number of papers to add to pending list')
    parser.add_argument('--clear-pending', action='store_true',
                        help='Clear the pending list before adding new papers')
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    text_dir = args.text_dir
    pending_file = args.pending_file
    dry_run = args.dry_run
    max_papers = args.max_papers
    clear_pending = args.clear_pending

    # Ensure the directories exist
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return
    if not os.path.exists(text_dir):
        logger.error(f"Text directory not found: {text_dir}")
        return

    # Create the parent directory for pending_file if it doesn't exist
    os.makedirs(os.path.dirname(pending_file), exist_ok=True)

    # Load existing pending papers, if any
    pending_papers = []
    if os.path.exists(pending_file) and not clear_pending:
        try:
            with open(pending_file, 'r') as f:
                pending_papers = json.load(f)
            logger.info(f"Loaded {len(pending_papers)} existing papers from pending list")
        except Exception as e:
            logger.error(f"Error loading pending papers: {e}")
    elif clear_pending:
        logger.info("Clearing existing pending list")

    # Get existing arxiv_ids in the pending list
    existing_ids = {paper.get('arxiv_id', '') for paper in pending_papers}

    # List all PDF files and text files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    text_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    logger.info(f"Found {len(text_files)} text files in {text_dir}")

    # Normalize text filenames for comparison
    text_base_names = {normalize_filename(f) for f in text_files}

    # Find PDFs that don't have corresponding text files
    missing_texts = []
    for pdf_file in pdf_files:
        normalized_pdf_name = normalize_filename(pdf_file)
        if normalized_pdf_name not in text_base_names:
            missing_texts.append(pdf_file)

    logger.info(f"Found {len(missing_texts)} PDFs without corresponding text files")

    # Limit the number of papers to add if max_papers is specified
    if max_papers is not None and max_papers < len(missing_texts):
        logger.info(f"Limiting to {max_papers} papers as specified")
        missing_texts = missing_texts[:max_papers]

    # Add missing PDFs to the pending list
    new_papers_count = 0
    for pdf_file in missing_texts:
        # Extract arxiv_id from filename (assuming format like 1234.56789v1.pdf or hash.pdf)
        pdf_base_name = os.path.splitext(pdf_file)[0]
        arxiv_id = pdf_base_name
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        # Skip if already in pending list
        if arxiv_id in existing_ids:
            logger.info(f"Paper {arxiv_id} already in pending list, skipping")
            continue
            
        # Create paper info
        paper_info = {
            'arxiv_id': arxiv_id,
            'title': f"Unknown Paper {arxiv_id}",
            'authors': [],
            'summary': "",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'categories': [],
            'pdf_url': "",
            'pdf_path': pdf_path
        }
        
        if not dry_run:
            pending_papers.append(paper_info)
        
        new_papers_count += 1
        
        # Log progress every 25 papers
        if new_papers_count % 25 == 0:
            logger.info(f"Added {new_papers_count} new papers to pending list so far")
    
    if not dry_run and new_papers_count > 0:
        # Save updated pending papers list
        with open(pending_file, 'w') as f:
            json.dump(pending_papers, f, indent=2)
        
        logger.info(f"Added {new_papers_count} new papers to pending list")
        logger.info(f"Total papers in pending list: {len(pending_papers)}")
        logger.info(f"To process these papers, run: python arxiv_collector.py --batch-process")
    elif dry_run:
        logger.info(f"Dry run: Would add {new_papers_count} papers to pending list")
    else:
        logger.info("No new papers added to pending list")

if __name__ == "__main__":
    main() 