"""
PDF document extractor module for legal contract analysis.
Handles text extraction and basic structure analysis from PDF files.
"""

import pdfplumber
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts text and structure from PDF documents."""

    def __init__(self):
        """Initialize the PDF extractor."""
        self.section_patterns = [
            r'^(\d+\.\s*[A-Z][^\n]+)',  # Numbered sections (e.g., "1. Introduction")
            r'^([A-Z][A-Z\s]+:)',  # ALL CAPS sections (e.g., "ARTICLE I:")
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)',  # Title Case sections (e.g., "Payment Terms:")
        ]

    def extract_text(self, file_path: str) -> Dict[str, any]:
        """
        Extract text and metadata from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing:
            - text: Full document text
            - metadata: Document metadata
            - sections: Identified sections
            - pages: List of page texts
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if file_path.suffix.lower() != '.pdf':
                raise ValueError(f"Invalid file type: {file_path}. Expected .pdf")

            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                metadata = self._extract_metadata(pdf)

                # Extract text from each page
                pages = []
                full_text = []

                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                    full_text.append(page_text)

                # Combine all text
                text = "\n".join(full_text)

                # Identify sections
                sections = self._identify_sections(text)

                return {
                    "text": text,
                    "metadata": metadata,
                    "sections": sections,
                    "pages": pages,
                    "total_pages": len(pdf.pages)
                }

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def _extract_metadata(self, pdf: pdfplumber.PDF) -> Dict[str, any]:
        """
        Extract metadata from PDF document.

        Args:
            pdf: pdfplumber PDF object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        try:
            if pdf.metadata:
                metadata.update({
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "producer": pdf.metadata.get("Producer", ""),
                    "creation_date": pdf.metadata.get("CreationDate", ""),
                    "modification_date": pdf.metadata.get("ModDate", "")
                })
                # Convert dates to strings for JSON serialization
                if isinstance(metadata["creation_date"], datetime):
                    metadata["creation_date"] = metadata["creation_date"].isoformat()
                if isinstance(metadata["modification_date"], datetime):
                    metadata["modification_date"] = metadata["modification_date"].isoformat()
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")

        return metadata

    def _identify_sections(self, text: str) -> List[Dict[str, any]]:
        """
        Identify document sections using pattern matching.

        Args:
            text: Document text

        Returns:
            List of identified sections with their text
        """
        sections = []
        lines = text.split("\n")

        current_section = None
        current_text = []

        for line in lines:
            is_section = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    if current_section:
                        sections.append({
                            "title": current_section,
                            "text": "\n".join(current_text).strip()
                        })
                    current_section = match.group(1).strip()
                    current_text = [line]
                    is_section = True
                    break
            if not is_section and current_section:
                current_text.append(line)

        # Add last section
        if current_section:
            sections.append({
                "title": current_section,
                "text": "\n".join(current_text).strip()
            })

        return sections

    def extract_tables(self, file_path: str) -> List[Dict[str, any]]:
        """
        Extract tables from PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of extracted tables
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()

                    for table_num, table in enumerate(page_tables, 1):
                        # Clean table data
                        cleaned_table = [[cell if cell is not None else "" for cell in row] for row in table]
                        if cleaned_table and any(cell for row in cleaned_table for cell in row):
                            tables.append({
                                "page": page_num,
                                "table_number": table_num,
                                "rows": len(cleaned_table),
                                "columns": len(cleaned_table[0]) if cleaned_table else 0,
                                "data": cleaned_table
                            })

            return tables

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {file_path}: {str(e)}")
            raise

    def extract_images(self, file_path: str, output_dir: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Extract images from PDF document.

        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save extracted images

        Returns:
            List of extracted image information
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            images = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    for image_num, image in enumerate(page.images, 1):
                        # Infer format (pdfplumber doesn't provide format directly)
                        image_data = image["stream"].get_data()
                        format_guess = "png"  # Default to PNG
                        if image_data.startswith(b"\xff\xd8"):
                            format_guess = "jpeg"
                        elif image_data.startswith(b"\x89PNG"):
                            format_guess = "png"

                        image_info = {
                            "page": page_num,
                            "image_number": image_num,
                            "width": image["width"],
                            "height": image["height"],
                            "format": format_guess
                        }

                        if output_dir:
                            output_path = Path(output_dir) / f"page_{page_num}_image_{image_num}.{format_guess}"
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            image_info["file_path"] = str(output_path)

                        images.append(image_info)

            return images

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting images from PDF {file_path}: {str(e)}")
            raise