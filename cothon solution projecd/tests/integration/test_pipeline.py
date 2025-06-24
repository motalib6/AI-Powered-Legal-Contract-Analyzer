"""
Integration tests for the complete contract analysis pipeline.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json
import sys
from typing import Dict, List, Tuple
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.main import ContractAnalyzer
from src.extractors.pdf_extractor import PDFExtractor
from src.extractors.docx_extractor import DOCXExtractor
from src.models.ner_model import LegalNERModel
from src.models.classifier import ClauseClassifier
from src.models.summarizer import ContractSummarizer

class TestAnalysisPipeline(unittest.TestCase):
    """Test cases for the complete analysis pipeline."""
    
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
        
        4. CONFIDENTIALITY
        The parties agree to keep all information confidential.
        
        5. INDEMNIFICATION
        Party A shall indemnify Party B against all claims.
        """
        
        # Save sample text to both PDF and DOCX (mock)
        cls.sample_pdf = cls.test_dir / "sample.pdf"
        cls.sample_docx = cls.test_dir / "sample.docx"
        
        with open(cls.sample_pdf, "w") as f:
            f.write(cls.sample_text)
        
        with open(cls.sample_docx, "w") as f:
            f.write(cls.sample_text)
        
        # Create sample training data
        cls.training_data = {
            "ner": [
                {
                    "text": "This Agreement is made between ABC Corp and XYZ Ltd.",
                    "entities": [
                        (25, 33, "PARTY"),
                        (38, 45, "PARTY")
                    ]
                }
            ],
            "classifier": [
                {
                    "text": "The parties agree to keep all information confidential.",
                    "label": "CONFIDENTIALITY"
                }
            ],
            "summarizer": [
                {
                    "text": cls.sample_text,
                    "summary": "Service agreement with confidentiality and indemnification clauses."
                }
            ]
        }
        
        # Save training data
        for model_type, data in cls.training_data.items():
            with open(cls.test_dir / f"{model_type}_train.json", "w") as f:
                json.dump(data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DOCXExtractor()
        
        # Initialize and train models
        self.ner_model = LegalNERModel()
        self.ner_model.train(
            train_data=self.training_data["ner"],
            output_dir=str(self.test_dir / "ner_model"),
            iterations=2,
            dropout=0.1
        )
        
        self.classifier = ClauseClassifier(model_type="tfidf")
        self.classifier.train(
            train_data=self.training_data["classifier"],
            output_dir=str(self.test_dir / "classifier_model"),
            test_size=0.2,
            random_state=42
        )
        
        self.summarizer = ContractSummarizer()
        self.summarizer.train(
            train_data=self.training_data["summarizer"],
            output_dir=str(self.test_dir / "summarizer_model"),
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-5
        )
        
        # Create analyzer with trained models
        self.analyzer = ContractAnalyzer()
        self.analyzer.ner_model = self.ner_model
        self.analyzer.classifier = self.classifier
        self.analyzer.summarizer = self.summarizer
    
    def test_pdf_pipeline(self):
        """Test complete pipeline with PDF input."""
        # Analyze PDF
        results = self.analyzer.analyze_contract(str(self.sample_pdf))
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn("summary", results)
        self.assertIn("clauses", results)
        self.assertIn("entities", results)
        self.assertIn("risks", results)
        
        # Check summary
        self.assertIsInstance(results["summary"], str)
        self.assertTrue(len(results["summary"]) > 0)
        
        # Check clauses
        self.assertTrue(len(results["clauses"]) > 0)
        for clause in results["clauses"]:
            self.assertIn("type", clause)
            self.assertIn("text", clause)
            self.assertIn("confidence", clause)
        
        # Check entities
        self.assertTrue(len(results["entities"]) > 0)
        for entity_type, entities in results["entities"].items():
            self.assertIsInstance(entities, list)
            for entity in entities:
                self.assertIn("text", entity)
                self.assertIn("confidence", entity)
        
        # Check risks
        self.assertTrue(len(results["risks"]) > 0)
        for risk in results["risks"]:
            self.assertIn("type", risk)
            self.assertIn("description", risk)
            self.assertIn("severity", risk)
    
    def test_docx_pipeline(self):
        """Test complete pipeline with DOCX input."""
        # Analyze DOCX
        results = self.analyzer.analyze_contract(str(self.sample_docx))
        
        # Check results structure (same as PDF)
        self.assertIsInstance(results, dict)
        self.assertIn("summary", results)
        self.assertIn("clauses", results)
        self.assertIn("entities", results)
        self.assertIn("risks", results)
        
        # Check summary
        self.assertIsInstance(results["summary"], str)
        self.assertTrue(len(results["summary"]) > 0)
        
        # Check clauses
        self.assertTrue(len(results["clauses"]) > 0)
        for clause in results["clauses"]:
            self.assertIn("type", clause)
            self.assertIn("text", clause)
            self.assertIn("confidence", clause)
        
        # Check entities
        self.assertTrue(len(results["entities"]) > 0)
        for entity_type, entities in results["entities"].items():
            self.assertIsInstance(entities, list)
            for entity in entities:
                self.assertIn("text", entity)
                self.assertIn("confidence", entity)
        
        # Check risks
        self.assertTrue(len(results["risks"]) > 0)
        for risk in results["risks"]:
            self.assertIn("type", risk)
            self.assertIn("description", risk)
            self.assertIn("severity", risk)
    
    def test_model_consistency(self):
        """Test consistency of model predictions."""
        # Analyze both PDF and DOCX
        pdf_results = self.analyzer.analyze_contract(str(self.sample_pdf))
        docx_results = self.analyzer.analyze_contract(str(self.sample_docx))
        
        # Compare summaries
        self.assertEqual(pdf_results["summary"], docx_results["summary"])
        
        # Compare clause types
        pdf_clause_types = {c["type"] for c in pdf_results["clauses"]}
        docx_clause_types = {c["type"] for c in docx_results["clauses"]}
        self.assertEqual(pdf_clause_types, docx_clause_types)
        
        # Compare entity types
        pdf_entity_types = set(pdf_results["entities"].keys())
        docx_entity_types = set(docx_results["entities"].keys())
        self.assertEqual(pdf_entity_types, docx_entity_types)
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with empty file
        empty_file = self.test_dir / "empty.pdf"
        empty_file.touch()
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_contract(str(empty_file))
        
        # Test with invalid file
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_contract("nonexistent.pdf")
        
        # Test with unsupported file type
        unsupported_file = self.test_dir / "test.txt"
        unsupported_file.touch()
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_contract(str(unsupported_file))

if __name__ == "__main__":
    unittest.main() 