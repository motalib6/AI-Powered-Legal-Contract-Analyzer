"""
Clause classifier module for legal contract analysis.
Uses scikit-learn and transformers for clause classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClauseClassifier:
    """Classifier for legal contract clauses."""

    # Define clause types
    CLAUSE_TYPES = {
        "INDEMNITY": "Indemnification and liability clauses",
        "CONFIDENTIALITY": "Confidentiality and non-disclosure clauses",
        "TERMINATION": "Termination and cancellation clauses",
        "GOVERNING_LAW": "Governing law and jurisdiction clauses",
        "INTELLECTUAL_PROPERTY": "IP rights and ownership clauses",
        "WARRANTY": "Warranty and representation clauses",
        "LIMITATION_OF_LIABILITY": "Limitation of liability clauses",
        "FORCE_MAJEURE": "Force majeure and act of god clauses",
        "ASSIGNMENT": "Assignment and transfer clauses",
        "AMENDMENT": "Amendment and modification clauses",
        "NOTICE": "Notice and communication clauses",
        "SEVERABILITY": "Severability and invalidity clauses",
        "ENTIRE_AGREEMENT": "Entire agreement and merger clauses",
        "WAIVER": "Waiver and non-waiver clauses",
        "OTHER": "Other types of clauses"
    }

    def __init__(self, model_type: str = "tfidf", model_path: Optional[str] = None):
        """
        Initialize the clause classifier.

        Args:
            model_type: Type of model to use ("tfidf" or "transformer")
            model_path: Path to a pre-trained model (optional)
        """
        self.model_type = model_type.lower()
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.transformer_model = None

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on type."""
        try:
            if self.model_type == "tfidf":
                # TF-IDF + Logistic Regression pipeline
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english"
                )
                self.model = Pipeline([
                    ("vectorizer", self.vectorizer),
                    ("classifier", LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1
                    ))
                ])

            elif self.model_type == "transformer":
                # Legal-BERT model
                self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
                self.transformer_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
                self.model = LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1
                )

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            logger.info(f"Initialized {self.model_type} model")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def train(self, training_data: List[Dict], output_dir: str,
              test_size: float = 0.2, random_state: int = 42):
        """
        Train the clause classifier.

        Args:
            training_data: List of training examples
            output_dir: Directory to save the trained model
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        try:
            # Prepare data
            texts = [example["text"] for example in training_data]
            labels = [example["label"] for example in training_data]

            # Validate label distribution
            label_counts = Counter(labels)
            missing_labels = set(self.CLAUSE_TYPES.keys()) - set(label_counts.keys())
            if missing_labels:
                logger.warning(f"Missing labels in training data: {missing_labels}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels if len(set(labels)) > 1 else None
            )

            if self.model_type == "tfidf":
                # Train TF-IDF model
                self.model.fit(X_train, y_train)

                # Evaluate
                y_pred = self.model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

            elif self.model_type == "transformer":
                # Prepare transformer features
                X_train_features = self._get_transformer_features(X_train)
                X_test_features = self._get_transformer_features(X_test)

                # Train classifier
                self.model.fit(X_train_features, y_train)

                # Evaluate
                y_pred = self.model.predict(X_test_features)
                report = classification_report(y_test, y_pred, output_dict=True)

            # Save model
            self.save(output_dir)

            # Save evaluation report
            report_path = Path(output_dir) / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Model trained and saved to {output_dir}")
            logger.info(f"Evaluation report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict clause type and confidence.

        Args:
            text: Input text

        Returns:
            Dictionary containing predicted label and confidence
        """
        try:
            if self.model_type == "tfidf":
                # Get prediction probabilities
                probs = self.model.predict_proba([text])[0]
                label_idx = np.argmax(probs)
                confidence = probs[label_idx]
                label = self.model.classes_[label_idx]

            elif self.model_type == "transformer":
                # Get transformer features
                features = self._get_transformer_features([text])

                # Get prediction probabilities
                probs = self.model.predict_proba(features)[0]
                label_idx = np.argmax(probs)
                confidence = probs[label_idx]
                label = self.model.classes_[label_idx]

            return {
                "label": label,
                "confidence": float(confidence),
                "description": self.CLAUSE_TYPES.get(label, "")
            }

        except Exception as e:
            logger.error(f"Error predicting clause type: {str(e)}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict clause types for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of predictions
        """
        try:
            if self.model_type == "tfidf":
                # Get prediction probabilities
                probs = self.model.predict_proba(texts)
                labels = self.model.classes_[np.argmax(probs, axis=1)]
                confidences = np.max(probs, axis=1)

            elif self.model_type == "transformer":
                # Get transformer features
                features = self._get_transformer_features(texts)

                # Get prediction probabilities
                probs = self.model.predict_proba(features)
                labels = self.model.classes_[np.argmax(probs, axis=1)]
                confidences = np.max(probs, axis=1)

            # Format results
            results = []
            for label, confidence in zip(labels, confidences):
                results.append({
                    "label": label,
                    "confidence": float(confidence),
                    "description": self.CLAUSE_TYPES.get(label, "")
                })

            return results

        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            raise

    def _get_transformer_features(self, texts: List[str]) -> np.ndarray:
        """
        Get transformer model features for texts.

        Args:
            texts: List of input texts

        Returns:
            Array of features
        """
        try:
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move encodings to the same device as the model
            device = next(self.transformer_model.parameters()).device
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Get transformer outputs
            with torch.no_grad():
                outputs = self.transformer_model(**encodings)
                # Use [CLS] token embeddings as features
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return features

        except Exception as e:
            logger.error(f"Error getting transformer features: {str(e)}")
            raise

    def save(self, output_dir: str):
        """
        Save the model to disk.

        Args:
            output_dir: Directory to save the model
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if self.model_type == "tfidf":
                # Save pipeline
                joblib.dump(self.model, output_path / "model.pkl")

            elif self.model_type == "transformer":
                # Save transformer model and tokenizer
                self.transformer_model.save_pretrained(output_path / "transformer")
                self.tokenizer.save_pretrained(output_path / "tokenizer")

                # Save classifier
                joblib.dump(self.model, output_path / "classifier.pkl")

            # Save clause type descriptions
            with open(output_path / "clause_types.json", "w") as f:
                json.dump(self.CLAUSE_TYPES, f, indent=2)

            logger.info(f"Model saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, model_path: str) -> "ClauseClassifier":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded ClauseClassifier instance
        """
        try:
            model_path = Path(model_path)

            # Load clause type descriptions
            with open(model_path / "clause_types.json") as f:
                clause_types = json.load(f)

            # Determine model type
            if (model_path / "model.pkl").exists():
                model_type = "tfidf"
            elif (model_path / "transformer").exists():
                model_type = "transformer"
            else:
                raise ValueError("Could not determine model type")

            # Create instance
            model = cls(model_type=model_type)
            model.CLAUSE_TYPES = clause_types

            if model_type == "tfidf":
                # Load pipeline
                model.model = joblib.load(model_path / "model.pkl")
                model.vectorizer = model.model.named_steps["vectorizer"]

            elif model_type == "transformer":
                # Load transformer model and tokenizer
                model.transformer_model = AutoModel.from_pretrained(model_path / "transformer")
                model.tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")

                # Load classifier
                model.model = joblib.load(model_path / "classifier.pkl")

            logger.info(f"Loaded {model_type} model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise