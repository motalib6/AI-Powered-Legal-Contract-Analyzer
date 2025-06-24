"""
Unit tests for AI models.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json
from typing import Dict, List, Tuple
import torch
import numpy as np

from src.models.ner_model import LegalNERModel
from src.models.classifier import ClauseClassifier
from src.models.summarizer import ContractSummarizer

class TestLegalNERModel(unittest.TestCase):
    """Test cases for LegalNERModel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample training data
        cls.sample_data = [
            {
                "text": "This Agreement is made between ABC Corp and XYZ Ltd.",
                "entities": [
                    (25, 33, "PARTY"),
                    (38, 45, "PARTY")
                ]
            },
            {
                "text": "The contract shall terminate on December 31, 2024.",
                "entities": [
                    (31, 45, "DATE")
                ]
            }
        ]
        
        # Save sample data
        with open(cls.test_dir / "sample_ner_data.json", "w") as f:
            json.dump(cls.sample_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        self.model = LegalNERModel()
    
    def test_train(self):
        """Test model training."""
        try:
            # Train model
            self.model.train(
                train_data=self.sample_data,
                output_dir=str(self.test_dir / "ner_model"),
                iterations=2,  # Small number for testing
                dropout=0.1
            )
            
            # Check if model files exist
            self.assertTrue((self.test_dir / "ner_model" / "model").exists())
            self.assertTrue((self.test_dir / "ner_model" / "config.json").exists())
            
        except Exception as e:
            self.fail(f"Training failed: {str(e)}")
    
    def test_predict(self):
        """Test entity prediction."""
        # Train model first
        self.model.train(
            train_data=self.sample_data,
            output_dir=str(self.test_dir / "ner_model"),
            iterations=2,
            dropout=0.1
        )
        
        # Test prediction
        text = "The agreement between Company A and Company B expires on January 1, 2025."
        entities = self.model.predict(text)
        
        self.assertIsInstance(entities, list)
        self.assertTrue(len(entities) > 0)
        
        # Check entity structure
        for entity in entities:
            self.assertIn("text", entity)
            self.assertIn("type", entity)
            self.assertIn("start", entity)
            self.assertIn("end", entity)
            self.assertIn("confidence", entity)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train model first
        self.model.train(
            train_data=self.sample_data,
            output_dir=str(self.test_dir / "ner_model"),
            iterations=2,
            dropout=0.1
        )
        
        # Evaluate model
        metrics = self.model.evaluate(self.sample_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check metric values
        for metric in metrics.values():
            self.assertIsInstance(metric, float)
            self.assertTrue(0 <= metric <= 1)

class TestClauseClassifier(unittest.TestCase):
    """Test cases for ClauseClassifier."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample training data
        cls.sample_data = [
            {
                "text": "The parties agree to keep all information confidential.",
                "label": "CONFIDENTIALITY"
            },
            {
                "text": "Party A shall indemnify Party B against all claims.",
                "label": "INDEMNITY"
            }
        ]
        
        # Save sample data
        with open(cls.test_dir / "sample_classifier_data.json", "w") as f:
            json.dump(cls.sample_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        self.model = ClauseClassifier(model_type="tfidf")  # Use TF-IDF for faster testing
    
    def test_train(self):
        """Test model training."""
        try:
            # Train model
            self.model.train(
                train_data=self.sample_data,
                output_dir=str(self.test_dir / "classifier_model"),
                test_size=0.2,
                random_state=42
            )
            
            # Check if model files exist
            self.assertTrue((self.test_dir / "classifier_model" / "model.pkl").exists())
            self.assertTrue((self.test_dir / "classifier_model" / "vectorizer.pkl").exists())
            
        except Exception as e:
            self.fail(f"Training failed: {str(e)}")
    
    def test_predict(self):
        """Test clause classification."""
        # Train model first
        self.model.train(
            train_data=self.sample_data,
            output_dir=str(self.test_dir / "classifier_model"),
            test_size=0.2,
            random_state=42
        )
        
        # Test prediction
        text = "The parties shall maintain the confidentiality of all proprietary information."
        prediction = self.model.predict(text)
        
        self.assertIsInstance(prediction, dict)
        self.assertIn("type", prediction)
        self.assertIn("confidence", prediction)
        
        # Check prediction values
        self.assertIsInstance(prediction["type"], str)
        self.assertIsInstance(prediction["confidence"], float)
        self.assertTrue(0 <= prediction["confidence"] <= 1)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        # Train model first
        self.model.train(
            train_data=self.sample_data,
            output_dir=str(self.test_dir / "classifier_model"),
            test_size=0.2,
            random_state=42
        )
        
        # Test batch prediction
        texts = [
            "The parties agree to keep all information confidential.",
            "Party A shall indemnify Party B against all claims."
        ]
        predictions = self.model.predict_batch(texts)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(texts))
        
        # Check prediction structure
        for pred in predictions:
            self.assertIsInstance(pred, dict)
            self.assertIn("type", pred)
            self.assertIn("confidence", pred)

class TestContractSummarizer(unittest.TestCase):
    """Test cases for ContractSummarizer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample training data
        cls.sample_data = [
            {
                "text": """
                This Agreement is made between Party A and Party B.
                Party A agrees to provide services to Party B.
                Party B agrees to pay Party A $10,000 per month.
                The agreement shall terminate on December 31, 2024.
                """,
                "summary": "Service agreement between Party A and Party B with monthly payment of $10,000, terminating on December 31, 2024."
            }
        ]
        
        # Save sample data
        with open(cls.test_dir / "sample_summarizer_data.json", "w") as f:
            json.dump(cls.sample_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        self.model = ContractSummarizer()
    
    def test_summarize(self):
        """Test contract summarization."""
        # Test summarization
        text = """
        This Agreement is made between Company A and Company B.
        Company A shall provide consulting services to Company B.
        Company B shall pay Company A $5,000 per month.
        The agreement may be terminated with 30 days notice.
        """
        
        summary = self.model.summarize(text)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("summary", summary)
        self.assertIn("metadata", summary)
        
        # Check summary content
        self.assertIsInstance(summary["summary"], str)
        self.assertTrue(len(summary["summary"]) < len(text))
        
        # Check metadata
        metadata = summary["metadata"]
        self.assertIn("original_length", metadata)
        self.assertIn("summary_length", metadata)
        self.assertIn("compression_ratio", metadata)
    
    def test_summarize_batch(self):
        """Test batch summarization."""
        # Test batch summarization
        texts = [
            """
            This Agreement is made between Party A and Party B.
            Party A agrees to provide services to Party B.
            Party B agrees to pay Party A $10,000 per month.
            """,
            """
            This Agreement is made between Company X and Company Y.
            Company X shall deliver products to Company Y.
            Company Y shall pay Company X $20,000 per delivery.
            """
        ]
        
        summaries = self.model.summarize_batch(texts)
        
        self.assertIsInstance(summaries, list)
        self.assertEqual(len(summaries), len(texts))
        
        # Check summary structure
        for summary in summaries:
            self.assertIsInstance(summary, dict)
            self.assertIn("summary", summary)
            self.assertIn("metadata", summary)

if __name__ == "__main__":
    unittest.main() 