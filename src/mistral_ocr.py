"""
Mistral OCR Module for extracting text from PDFs using Mistral AI's API
"""

import os
import base64
import logging
import time
import re
import json
import requests
from typing import Tuple, Optional
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Try to import config, but don't fail if it's not available
try:
    import config
    HAS_CONFIG = True
except ImportError:
    logger.warning("config module not available")
    HAS_CONFIG = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralOCR:
    """
    Class for extracting text from PDFs using Mistral AI's OCR API
    """
    
    def __init__(self, api_key=None, model=None):
        """
        Initialize MistralOCR
        
        Args:
            api_key: Mistral API key (optional, can be loaded from config)
            model: Model to use (optional, can be loaded from config)
        """
        self.api_key = api_key
        self.model = model or "mistral-ocr-latest"
        
        # Load from config if available
        if not self.api_key and HAS_CONFIG:
            if hasattr(config, 'MISTRAL_API_KEY'):
                self.api_key = config.MISTRAL_API_KEY
            elif hasattr(config, 'MISTRAL_OCR_API_KEY'):
                self.api_key = config.MISTRAL_OCR_API_KEY
                
        if not self.model and HAS_CONFIG and hasattr(config, 'MISTRAL_OCR_MODEL'):
            self.model = config.MISTRAL_OCR_MODEL
        
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI SDK is not installed. Please install it with 'pip install mistralai'")
            
        self.client = Mistral(api_key=self.api_key)
        logger.info(f"Initialized MistralOCR with model: {self.model}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from a PDF file using Mistral AI's OCR API
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            tuple: (extracted_text, title)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        
        # If file is small enough (less than 2MB), use direct base64 encoding
        if file_size < 2 * 1024 * 1024:  # 2MB
            return self._extract_text_with_direct_method(pdf_path)
        else:
            # For larger files, use file upload method
            return self._extract_text_with_file_upload(pdf_path)
    
    def _extract_text_with_direct_method(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text using direct base64 encoding (for smaller PDFs)
        """
        try:
            # Read the PDF file
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
                
            # Encode the PDF content as base64
            pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
            data_url = f"data:application/pdf;base64,{pdf_base64}"
            
            # Process the PDF with OCR API
            logger.info(f"Calling Mistral OCR API (direct method) for PDF: {os.path.basename(pdf_path)}")
            
            # Using the dedicated OCR API
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": data_url
                }
            )
            
            return self._process_ocr_response(ocr_response, pdf_path)
            
        except Exception as e:
            logger.error(f"Error extracting text with direct method: {str(e)}")
            logger.exception("Full error traceback:")
            return "", None
    
    def _extract_text_with_file_upload(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text using file upload approach (for larger PDFs)
        """
        try:
            # File upload approach
            logger.info(f"Using file upload method for large PDF: {os.path.basename(pdf_path)}")
            
            # Step 1: Upload the file
            with open(pdf_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(pdf_path),
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            logger.info(f"File uploaded with ID: {uploaded_file.id}")
            
            # Step 2: Get a signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Step 3: Process the file using OCR
            logger.info(f"Processing uploaded file with OCR API: {os.path.basename(pdf_path)}")
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
            
            return self._process_ocr_response(ocr_response, pdf_path)
            
        except Exception as e:
            logger.error(f"Error extracting text with file upload method: {str(e)}")
            logger.exception("Full error traceback:")
            return self.extract_text_using_chat_api(pdf_path)
    
    def _process_ocr_response(self, ocr_response, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Process OCR response to extract text and title
        """
        # Combine text from all pages
        full_text = ""
        for page in ocr_response.pages:
            full_text += page.markdown + "\n\n"
            
        logger.info(f"Successfully extracted text from PDF: {os.path.basename(pdf_path)} ({len(ocr_response.pages)} pages)")
        
        # Try to extract title from first page content
        title = None
        if ocr_response.pages:
            first_page = ocr_response.pages[0].markdown
            # Look for patterns like "# Title" or first non-empty line
            title_match = re.search(r"^#\s+(.+)$", first_page, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Take first non-empty line as title
                for line in first_page.split('\n'):
                    line = line.strip()
                    if line and len(line) > 5 and len(line) < 200:
                        title = line
                        break
        
        return full_text, title
    
    def extract_text_with_fallback(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from PDF with fallback to chat API if OCR API fails
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            tuple: (extracted_text, title)
        """
        try:
            # Try the OCR API first
            return self.extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"OCR API failed, trying fallback method: {str(e)}")
            return self.extract_text_using_chat_api(pdf_path)
            
    def extract_text_using_chat_api(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """
        Fallback method to extract text using the chat API
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            tuple: (extracted_text, title)
        """
        try:
            # Check if file is too large for chat API
            file_size = os.path.getsize(pdf_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"PDF is too large for chat API ({file_size / (1024*1024):.2f} MB). Compressing or using PyPDF2 instead.")
                return "", None

            # Read the PDF file
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
                
            # Encode the PDF content as base64
            pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
            
            # Process with chat API as fallback
            logger.info(f"Using chat API fallback for PDF: {os.path.basename(pdf_path)}")
            chat_response = self.client.chat.complete(
                model="mistral-large-latest",  # Using large model for better extraction
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert OCR system that extracts text from PDF documents."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this PDF document. Return the full text content and identify the title. Format your response as: TITLE: [Document Title]\n\nCONTENT:\n[Full extracted text]"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_base64}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Extract the response text
            response_text = chat_response.choices[0].message.content
            
            # Extract title and content
            title_match = re.search(r"TITLE:\s*(.*?)(?:\n\n|\n(?=CONTENT))", response_text, re.DOTALL)
            title = title_match.group(1).strip() if title_match else None
            
            content_match = re.search(r"CONTENT:\s*(.*)", response_text, re.DOTALL)
            content = content_match.group(1).strip() if content_match else response_text
            
            logger.info(f"Successfully extracted text using chat API fallback: {os.path.basename(pdf_path)}")
            return content, title
            
        except Exception as e:
            logger.error(f"Error extracting text with chat API fallback: {str(e)}")
            return "", None

    def extract_text(self, pdf_path: str, api_key: str = None, model: str = None) -> Tuple[bool, str]:
        """
        Extract text from a PDF file using Mistral OCR API
        
        Args:
            pdf_path: Path to the PDF file
            api_key: Mistral API key (optional, falls back to instance api_key)
            model: Model to use (optional, falls back to instance model)
            
        Returns:
            Tuple of success flag and text
        """
        api_key = api_key or self.api_key
        model = model or self.model
            
        if not api_key:
            logger.error("No Mistral API key provided")
            return False, ""
            
        return extract_text(pdf_path, api_key, model)

def get_mistral_ocr() -> Optional[MistralOCR]:
    """
    Create and return a MistralOCR instance if dependencies are available
    
    Returns:
        MistralOCR instance or None
    """
    try:
        # Check if Mistral OCR should be used
        if HAS_CONFIG and hasattr(config, 'USE_MISTRAL_OCR') and not config.USE_MISTRAL_OCR:
            logger.info("Mistral OCR is disabled in config")
            return None
            
        return MistralOCR()
    except Exception as e:
        logger.warning(f"Could not initialize Mistral OCR: {e}")
        return None

# Initialize MistralOCR if available
mistral_ocr = get_mistral_ocr()

def extract_text_from_pdf_with_mistral(pdf_path, model="mistral-large-pdf", api_key=None, max_retries=3, sleep_time=2):
    """
    Extract text from a PDF file using Mistral OCR
    
    Args:
        pdf_path: Path to the PDF file
        model: Mistral model to use for OCR
        api_key: Mistral API key
        max_retries: Maximum number of retries
        sleep_time: Time to sleep between retries
        
    Returns:
        str: Extracted text or None if extraction failed
    """
    if not api_key:
        logger.error("Mistral API key not provided")
        return None
        
    # Check file size
    file_size = os.path.getsize(pdf_path)
    if file_size > 10 * 1024 * 1024:  # If file is larger than 10MB
        logger.info(f"PDF is large ({file_size/1024/1024:.2f}MB), processing by pages")
        return process_large_pdf(pdf_path, model, api_key, max_retries, sleep_time)
    
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "content": {
                "document": base64_pdf,
                "mime_type": "application/pdf"
            },
            "parameters": {
                "format": "text"
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.mistral.ai/v1/ocr",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json.get("text", "")
                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API error occurred: Status {response.status_code}")
                    logger.error(response.text)
                    if "request body did not fit into client body buffer" in response.text:
                        logger.info("PDF too large for direct processing, trying page-by-page approach")
                        return process_large_pdf(pdf_path, model, api_key, max_retries, sleep_time)
                    break
                    
            except Exception as e:
                logger.error(f"Error during OCR request: {e}")
                time.sleep(sleep_time)
                
        return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF with Mistral OCR: {e}")
        return None

def process_large_pdf(pdf_path, model, api_key, max_retries=3, sleep_time=2):
    """
    Process a large PDF by extracting text page by page
    
    Args:
        pdf_path: Path to the PDF file
        model: Mistral model to use for OCR
        api_key: Mistral API key
        max_retries: Maximum number of retries
        sleep_time: Time to sleep between retries
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        # Open the PDF
        pdf = PdfReader(pdf_path)
        num_pages = len(pdf.pages)
        logger.info(f"Processing PDF with {num_pages} pages")
        
        # Process in chunks of 10 pages maximum
        chunk_size = 10
        all_text = []
        
        for i in range(0, num_pages, chunk_size):
            end_idx = min(i + chunk_size, num_pages)
            logger.info(f"Processing pages {i+1}-{end_idx} of {num_pages}")
            
            # Create a new PDF with just these pages
            chunk_pdf_bytes = BytesIO()
            
            # Use an alternative approach with a temporary file for better reliability
            chunk_path = f"{pdf_path}.chunk_{i}-{end_idx}.pdf"
            
            try:
                # Extract pages using PyPDF2
                pdf_writer = PdfWriter()
                
                for page_num in range(i, end_idx):
                    pdf_writer.add_page(pdf.pages[page_num])
                
                with open(chunk_path, "wb") as f:
                    pdf_writer.write(f)
                
                # Process the chunk with Mistral OCR
                chunk_text = extract_text_from_pdf_with_mistral(
                    chunk_path, 
                    model=model, 
                    api_key=api_key, 
                    max_retries=max_retries, 
                    sleep_time=sleep_time
                )
                
                if chunk_text:
                    all_text.append(chunk_text)
                else:
                    logger.warning(f"Failed to extract text from pages {i+1}-{end_idx}")
                
                # Sleep to avoid rate limiting
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error processing pages {i+1}-{end_idx}: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
        
        return "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"Error processing large PDF: {e}")
        return None

def extract_text(pdf_path: str, api_key: str, model: str = "mistral-large-pdf") -> Tuple[bool, str]:
    """
    Extract text from a PDF file using Mistral OCR API
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Mistral API key
        model: Model to use
        
    Returns:
        Tuple of success flag and text
    """
    try:
        text = extract_text_from_pdf_with_mistral(
            pdf_path=pdf_path,
            model=model,
            api_key=api_key,
            max_retries=3,
            sleep_time=2
        )
        
        if text:
            return True, text
        else:
            logger.warning(f"Failed to extract text from {pdf_path} using Mistral OCR")
            return False, ""
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF with Mistral OCR: {e}")
        return False, "" 