"""
PDF text extraction module.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDF files."""
    
    def __init__(self):
        """Initialize the PDF extractor."""
        try:
            import PyPDF2
            import pdfplumber
            self.pdf_extractor = pdfplumber
            self.pdf_reader = PyPDF2
            logger.info("PDF extractor initialized successfully")
        except ImportError as e:
            logger.error(f"Error importing PDF libraries: {str(e)}")
            raise

    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF file."""
        try:
            text = ""
            # Try pdfplumber first for better text extraction
            with self.pdf_extractor.open(file_path) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                    except Exception as e:
                        logger.warning(f"Error extracting text from page: {str(e)}")
                        continue
            
            # If pdfplumber fails, try PyPDF2 as backup
            if not text.strip():
                with open(file_path, 'rb') as f:
                    pdf = self.pdf_reader.PdfReader(f)
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + " "
                        except Exception as e:
                            logger.warning(f"Error extracting text from page (PyPDF2): {str(e)}")
                            continue
            
            # Clean up text
            text = text.replace('\x00', '')  # Remove null bytes
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.replace(' .', '.').replace(' ,', ',')  # Fix punctuation
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None 