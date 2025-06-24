"""
Unit tests for document extractors.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json
from typing import Dict, List

from src.extractors.pdf_extractor import PDFExtractor
from src.extractors.docx_extractor import DOCXExtractor

class TestPDFExtractor(unittest.TestCase):
    """Test cases for PDFExtractor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample PDF content
        cls.sample_text = """
        CONTRACT AGREEMENT
        
        This Agreement is made on January 1, 2024, between:
        
        PARTY A: ABC Corporation
        PARTY B: XYZ Ltd.
        
        1. DEFINITIONS
        1.1 "Confidential Information" means...
        
        2. OBLIGATIONS
        2.1 Party A shall...
        2.2 Party B shall...
        
        3. TERMINATION
        This agreement may be terminated...
        """
        
        # Save sample text to PDF (mock)
        cls.sample_pdf = cls.test_dir / "sample.pdf"
        with open(cls.sample_pdf, "w") as f:
            f.write(cls.sample_text)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        self.extractor = PDFExtractor()
    
    def test_extract_text(self):
        """Test text extraction from PDF."""
        result = self.extractor.extract_text(str(self.sample_pdf))
        
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("metadata", result)
        self.assertIn("sections", result)
        self.assertIn("pages", result)
        
        # Check extracted text
        self.assertIn("CONTRACT AGREEMENT", result["text"])
        self.assertIn("PARTY A: ABC Corporation", result["text"])
        
        # Check sections
        sections = result["sections"]
        self.assertTrue(any(s["title"] == "1. DEFINITIONS" for s in sections))
        self.assertTrue(any(s["title"] == "2. OBLIGATIONS" for s in sections))
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        result = self.extractor._extract_metadata(str(self.sample_pdf))
        
        self.assertIsInstance(result, dict)
        self.assertIn("title", result)
        self.assertIn("author", result)
        self.assertIn("creation_date", result)
    
    def test_identify_sections(self):
        """Test section identification."""
        sections = self.extractor._identify_sections(self.sample_text)
        
        self.assertIsInstance(sections, list)
        self.assertTrue(len(sections) > 0)
        
        # Check section titles
        section_titles = [s["title"] for s in sections]
        self.assertIn("1. DEFINITIONS", section_titles)
        self.assertIn("2. OBLIGATIONS", section_titles)
        self.assertIn("3. TERMINATION", section_titles)
    
    def test_invalid_file(self):
        """Test handling of invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text("nonexistent.pdf")

class TestDOCXExtractor(unittest.TestCase):
    """Test cases for DOCXExtractor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample DOCX content
        cls.sample_text = """
        CONTRACT AGREEMENT
        
        This Agreement is made on January 1, 2024, between:
        
        PARTY A: ABC Corporation
        PARTY B: XYZ Ltd.
        
        1. DEFINITIONS
        1.1 "Confidential Information" means...
        
        2. OBLIGATIONS
        2.1 Party A shall...
        2.2 Party B shall...
        
        3. TERMINATION
        This agreement may be terminated...
        """
        
        # Save sample text to DOCX (mock)
        cls.sample_docx = cls.test_dir / "sample.docx"
        with open(cls.sample_docx, "w") as f:
            f.write(cls.sample_text)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        self.extractor = DOCXExtractor()
    
    def test_extract_text(self):
        """Test text extraction from DOCX."""
        result = self.extractor.extract_text(str(self.sample_docx))
        
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("metadata", result)
        self.assertIn("sections", result)
        self.assertIn("paragraphs", result)
        
        # Check extracted text
        self.assertIn("CONTRACT AGREEMENT", result["text"])
        self.assertIn("PARTY A: ABC Corporation", result["text"])
        
        # Check sections
        sections = result["sections"]
        self.assertTrue(any(s["title"] == "1. DEFINITIONS" for s in sections))
        self.assertTrue(any(s["title"] == "2. OBLIGATIONS" for s in sections))
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        result = self.extractor._extract_metadata(str(self.sample_docx))
        
        self.assertIsInstance(result, dict)
        self.assertIn("title", result)
        self.assertIn("author", result)
        self.assertIn("creation_date", result)
        self.assertIn("statistics", result)
    
    def test_identify_sections(self):
        """Test section identification."""
        sections = self.extractor._identify_sections(self.sample_text)
        
        self.assertIsInstance(sections, list)
        self.assertTrue(len(sections) > 0)
        
        # Check section titles
        section_titles = [s["title"] for s in sections]
        self.assertIn("1. DEFINITIONS", section_titles)
        self.assertIn("2. OBLIGATIONS", section_titles)
        self.assertIn("3. TERMINATION", section_titles)
    
    def test_extract_paragraphs(self):
        """Test paragraph extraction with formatting."""
        result = self.extractor.extract_text(str(self.sample_docx))
        paragraphs = result["paragraphs"]
        
        self.assertIsInstance(paragraphs, list)
        self.assertTrue(len(paragraphs) > 0)
        
        # Check paragraph structure
        for para in paragraphs:
            self.assertIn("text", para)
            self.assertIn("style", para)
            self.assertIn("alignment", para)
    
    def test_invalid_file(self):
        """Test handling of invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text("nonexistent.docx")

if __name__ == "__main__":
    unittest.main() 