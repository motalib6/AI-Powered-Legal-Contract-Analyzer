"""
Unit tests for the main application.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json
from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.main import ContractAnalyzer

class TestContractAnalyzer(unittest.TestCase):
    """Test cases for ContractAnalyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample contract content
        cls.sample_text = """
        CONTRACT AGREEMENT
        
        This Agreement is made on January 1, 2024, between:
        
        PARTY A: ABC Corporation
        PARTY B: XYZ Ltd.
        
        1. DEFINITIONS
        1.1 "Confidential Information" means any information disclosed by one party to the other.
        
        2. OBLIGATIONS
        2.1 Party A shall provide consulting services to Party B.
        2.2 Party B shall pay Party A $10,000 per month.
        
        3. TERMINATION
        This agreement may be terminated with 30 days written notice.
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
        # Mock the model components
        self.ner_model = MagicMock()
        self.classifier = MagicMock()
        self.summarizer = MagicMock()
        
        # Create analyzer with mocked components
        self.analyzer = ContractAnalyzer()
        self.analyzer.ner_model = self.ner_model
        self.analyzer.classifier = self.classifier
        self.analyzer.summarizer = self.summarizer
    
    def test_analyze_contract(self):
        """Test contract analysis."""
        # Mock model predictions
        self.ner_model.predict.return_value = [
            {
                "text": "ABC Corporation",
                "type": "PARTY",
                "start": 0,
                "end": 15,
                "confidence": 0.95
            },
            {
                "text": "XYZ Ltd.",
                "type": "PARTY",
                "start": 0,
                "end": 8,
                "confidence": 0.92
            }
        ]
        
        self.classifier.predict.return_value = {
            "type": "CONFIDENTIALITY",
            "confidence": 0.88
        }
        
        self.summarizer.summarize.return_value = {
            "summary": "Service agreement between ABC Corporation and XYZ Ltd.",
            "metadata": {
                "original_length": 500,
                "summary_length": 100,
                "compression_ratio": 0.2
            }
        }
        
        # Analyze contract
        results = self.analyzer.analyze_contract(str(self.sample_pdf))
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn("summary", results)
        self.assertIn("clauses", results)
        self.assertIn("entities", results)
        self.assertIn("risks", results)
        
        # Check summary
        self.assertEqual(
            results["summary"],
            "Service agreement between ABC Corporation and XYZ Ltd."
        )
        
        # Check entities
        self.assertIn("PARTY", results["entities"])
        self.assertEqual(len(results["entities"]["PARTY"]), 2)
        
        # Check clauses
        self.assertTrue(len(results["clauses"]) > 0)
        self.assertEqual(results["clauses"][0]["type"], "CONFIDENTIALITY")
    
    def test_load_models(self):
        """Test model loading."""
        with patch("src.models.ner_model.LegalNERModel") as mock_ner, \
             patch("src.models.classifier.ClauseClassifier") as mock_classifier, \
             patch("src.models.summarizer.ContractSummarizer") as mock_summarizer:
            
            # Create new analyzer (should load models)
            analyzer = ContractAnalyzer()
            
            # Check if models were initialized
            mock_ner.assert_called_once()
            mock_classifier.assert_called_once()
            mock_summarizer.assert_called_once()
    
    def test_invalid_file(self):
        """Test handling of invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_contract("nonexistent.pdf")
    
    def test_empty_file(self):
        """Test handling of empty file."""
        # Create empty file
        empty_file = self.test_dir / "empty.pdf"
        empty_file.touch()
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_contract(str(empty_file))

class TestStreamlitApp(unittest.TestCase):
    """Test cases for Streamlit application."""
    
    def setUp(self):
        """Set up each test case."""
        # Mock Streamlit functions
        self.st_mock = MagicMock()
        self.st_mock.file_uploader.return_value = None
        self.st_mock.spinner.return_value.__enter__.return_value = None
        self.st_mock.spinner.return_value.__exit__.return_value = None
        
        # Patch Streamlit
        self.st_patcher = patch("streamlit", self.st_mock)
        self.st_patcher.start()
    
    def tearDown(self):
        """Clean up after each test case."""
        self.st_patcher.stop()
    
    def test_file_upload(self):
        """Test file upload functionality."""
        # Mock file upload
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"test content"
        
        self.st_mock.file_uploader.return_value = mock_file
        
        # Run app
        from scripts.run_app import main
        main()
        
        # Check if file uploader was called
        self.st_mock.file_uploader.assert_called_once_with(
            "Upload Contract Document",
            type=["pdf", "docx"],
            help="Upload a PDF or DOCX file containing a legal contract"
        )
    
    def test_no_file_uploaded(self):
        """Test behavior when no file is uploaded."""
        # Run app
        from scripts.run_app import main
        main()
        
        # Check that analysis was not performed
        self.st_mock.spinner.assert_not_called()
    
    def test_error_handling(self):
        """Test error handling in the app."""
        # Mock file upload with error
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.side_effect = Exception("Test error")
        
        self.st_mock.file_uploader.return_value = mock_file
        
        # Run app
        from scripts.run_app import main
        main()
        
        # Check if error was displayed
        self.st_mock.error.assert_called_once_with("Error analyzing contract: Test error")

if __name__ == "__main__":
    unittest.main() 