"""
Tests for evaluating model performance.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.ner_model import LegalNERModel
from src.models.classifier import ClauseClassifier
from src.models.summarizer import ContractSummarizer

class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance evaluation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample training and test data
        cls.training_data = {
            "ner": [
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
            ],
            "classifier": [
                {
                    "text": "The parties agree to keep all information confidential.",
                    "label": "CONFIDENTIALITY"
                },
                {
                    "text": "Party A shall indemnify Party B against all claims.",
                    "label": "INDEMNITY"
                }
            ],
            "summarizer": [
                {
                    "text": """
                    This Agreement is made between Party A and Party B.
                    Party A shall provide consulting services to Party B.
                    Party B shall pay Party A $10,000 per month.
                    The agreement shall terminate on December 31, 2024.
                    """,
                    "summary": "Service agreement with monthly payment of $10,000, terminating on December 31, 2024."
                }
            ]
        }
        
        # Create test data with slight variations
        cls.test_data = {
            "ner": [
                {
                    "text": "This Agreement is made between DEF Corp and PQR Ltd.",
                    "entities": [
                        (25, 33, "PARTY"),
                        (38, 45, "PARTY")
                    ]
                },
                {
                    "text": "The contract shall end on January 1, 2025.",
                    "entities": [
                        (25, 39, "DATE")
                    ]
                }
            ],
            "classifier": [
                {
                    "text": "Both parties shall maintain confidentiality of all information.",
                    "label": "CONFIDENTIALITY"
                },
                {
                    "text": "Party B shall indemnify Party A against any claims.",
                    "label": "INDEMNITY"
                }
            ],
            "summarizer": [
                {
                    "text": """
                    This Agreement is made between Company X and Company Y.
                    Company X shall provide services to Company Y.
                    Company Y shall pay Company X $15,000 per month.
                    The agreement may be terminated with 30 days notice.
                    """,
                    "summary": "Service agreement with monthly payment of $15,000 and 30-day termination notice."
                }
            ]
        }
        
        # Save data
        for data_type in ["training", "test"]:
            data = getattr(cls, f"{data_type}_data")
            for model_type, model_data in data.items():
                with open(cls.test_dir / f"{model_type}_{data_type}.json", "w") as f:
                    json.dump(model_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test case."""
        # Initialize models
        self.ner_model = LegalNERModel()
        self.classifier = ClauseClassifier(model_type="tfidf")
        self.summarizer = ContractSummarizer()
        
        # Train models
        self.ner_model.train(
            train_data=self.training_data["ner"],
            output_dir=str(self.test_dir / "ner_model"),
            iterations=5,
            dropout=0.1
        )
        
        self.classifier.train(
            train_data=self.training_data["classifier"],
            output_dir=str(self.test_dir / "classifier_model"),
            test_size=0.2,
            random_state=42
        )
        
        self.summarizer.train(
            train_data=self.training_data["summarizer"],
            output_dir=str(self.test_dir / "summarizer_model"),
            num_epochs=2,
            batch_size=1,
            learning_rate=2e-5
        )
    
    def test_ner_performance(self):
        """Test NER model performance."""
        # Evaluate model
        metrics = self.ner_model.evaluate(self.test_data["ner"])
        
        # Check metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check metric values
        for metric in metrics.values():
            self.assertIsInstance(metric, float)
            self.assertTrue(0 <= metric <= 1)
        
        # Check entity-specific metrics
        self.assertIn("PARTY", metrics)
        self.assertIn("DATE", metrics)
        
        # Check minimum performance threshold
        self.assertGreater(metrics["f1"], 0.5)  # F1 score should be at least 0.5
    
    def test_classifier_performance(self):
        """Test classifier performance."""
        # Get predictions
        texts = [item["text"] for item in self.test_data["classifier"]]
        true_labels = [item["label"] for item in self.test_data["classifier"]]
        
        predictions = self.classifier.predict_batch(texts)
        pred_labels = [pred["type"] for pred in predictions]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="weighted"
        )
        
        # Check minimum performance threshold
        self.assertGreater(f1, 0.5)  # F1 score should be at least 0.5
        
        # Check confidence scores
        for pred in predictions:
            self.assertIn("confidence", pred)
            self.assertIsInstance(pred["confidence"], float)
            self.assertTrue(0 <= pred["confidence"] <= 1)
    
    def test_summarizer_performance(self):
        """Test summarizer performance."""
        # Get summaries
        texts = [item["text"] for item in self.test_data["summarizer"]]
        true_summaries = [item["summary"] for item in self.test_data["summarizer"]]
        
        summaries = self.summarizer.summarize_batch(texts)
        
        # Check summary structure
        for summary in summaries:
            self.assertIn("summary", summary)
            self.assertIn("metadata", summary)
            
            # Check metadata
            metadata = summary["metadata"]
            self.assertIn("original_length", metadata)
            self.assertIn("summary_length", metadata)
            self.assertIn("compression_ratio", metadata)
            
            # Check compression ratio
            self.assertTrue(0 < metadata["compression_ratio"] < 1)
        
        # Check summary quality
        for summary, true_summary in zip(summaries, true_summaries):
            # Check length
            self.assertLess(
                len(summary["summary"]),
                len(summary["metadata"]["original_length"])
            )
            
            # Check content overlap (simple word overlap)
            summary_words = set(summary["summary"].lower().split())
            true_words = set(true_summary.lower().split())
            overlap = len(summary_words.intersection(true_words)) / len(true_words)
            
            self.assertGreater(overlap, 0.3)  # At least 30% word overlap
    
    def test_model_robustness(self):
        """Test model robustness to input variations."""
        # Test with noisy input
        noisy_text = """
        This Agreement is made between ABC Corp and XYZ Ltd.
        There are some typos and extra spaces here.
        The contract shall terminate on December 31, 2024.
        """
        
        # NER should still identify entities
        ner_entities = self.ner_model.predict(noisy_text)
        self.assertTrue(len(ner_entities) > 0)
        
        # Classifier should still predict clause type
        classifier_pred = self.classifier.predict(noisy_text)
        self.assertIn("type", classifier_pred)
        self.assertIn("confidence", classifier_pred)
        
        # Summarizer should still generate summary
        summary = self.summarizer.summarize(noisy_text)
        self.assertIn("summary", summary)
        self.assertTrue(len(summary["summary"]) > 0)
    
    def test_model_efficiency(self):
        """Test model efficiency."""
        import time
        
        # Test processing time for NER
        start_time = time.time()
        self.ner_model.predict_batch([item["text"] for item in self.test_data["ner"]])
        ner_time = time.time() - start_time
        
        # Test processing time for classifier
        start_time = time.time()
        self.classifier.predict_batch([item["text"] for item in self.test_data["classifier"]])
        classifier_time = time.time() - start_time
        
        # Test processing time for summarizer
        start_time = time.time()
        self.summarizer.summarize_batch([item["text"] for item in self.test_data["summarizer"]])
        summarizer_time = time.time() - start_time
        
        # Check processing times (should be reasonable)
        self.assertLess(ner_time, 1.0)  # NER should process in under 1 second
        self.assertLess(classifier_time, 1.0)  # Classifier should process in under 1 second
        self.assertLess(summarizer_time, 5.0)  # Summarizer should process in under 5 seconds

if __name__ == "__main__":
    unittest.main() 