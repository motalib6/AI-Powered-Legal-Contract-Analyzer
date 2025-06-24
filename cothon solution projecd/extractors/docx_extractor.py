"""
DOCX text extraction module.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class DOCXExtractor:
    """Extract text from DOCX files."""
    
    def __init__(self):
        """Initialize the DOCX extractor."""
        try:
            from docx import Document
            self.Document = Document
            logger.info("DOCX extractor initialized successfully")
        except ImportError as e:
            logger.error(f"Error importing docx library: {str(e)}")
            raise

    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a DOCX file."""
        try:
            doc = self.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Clean up text
            text = text.replace('\x00', '')  # Remove null bytes
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.replace(' .', '.').replace(' ,', ',')  # Fix punctuation
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return None 