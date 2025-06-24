"""
Named Entity Recognition (NER) model for legal contract analysis.
Uses SpaCy for custom entity recognition of legal entities.
"""

import spacy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from spacy.tokens import Doc
from spacy.training import Example
import random
import spacy.cli

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalNERModel:
    """Custom NER model for legal contract analysis."""

    # Define legal entity types
    ENTITY_TYPES = {
        "PARTY": "Contracting parties (e.g., Buyer, Seller, Licensor)",
        "OBLIGATION": "Contractual obligations and duties",
        "CLAUSE_TYPE": "Types of clauses (e.g., Indemnity, Confidentiality)",
        "DATE": "Dates and time periods",
        "AMOUNT": "Monetary amounts and financial terms",
        "JURISDICTION": "Governing law and jurisdiction",
        "TERMINATION": "Termination conditions and clauses"
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the NER model.

        Args:
            model_path: Path to a pre-trained model (optional)
        """
        if model_path and Path(model_path).exists():
            try:
                self.nlp = spacy.load(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                self._initialize_base_model()
        else:
            self._initialize_base_model()

    def _initialize_base_model(self):
        """Initialize base model with custom entity types."""
        try:
            # Start with a blank English model
            self.nlp = spacy.blank("en")

            # Add NER pipeline
            if "ner" not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe("ner")

                # Add entity labels
                for label in self.ENTITY_TYPES.keys():
                    ner.add_label(label)

            logger.info("Initialized base model with custom entity types")

        except Exception as e:
            logger.error(f"Error initializing base model: {str(e)}")
            raise

    def train(self, training_data: List[Dict], output_dir: str,
              iterations: int = 30, dropout: float = 0.2):
        """
        Train the NER model on custom training data.

        Args:
            training_data: List of training examples
            output_dir: Directory to save the trained model
            iterations: Number of training iterations
            dropout: Dropout rate for training
        """
        try:
            # Convert training data to SpaCy format
            train_examples = []
            for example in training_data:
                text = example["text"]
                entities = example["entities"]

                # Create Doc object
                doc = self.nlp.make_doc(text)

                # Add entity annotations
                ents = []
                for start, end, label in entities:
                    span = doc.char_span(start, end, label=label)
                    if span:
                        ents.append(span)
                    else:
                        logger.warning(f"Invalid span for entity: {text[start:end]}")

                if ents:  # Only add examples with valid entities
                    doc.ents = ents
                    train_examples.append(
                        Example.from_dict(doc, {"entities": [(e.start, e.end, e.label_) for e in doc.ents]}))

            if not train_examples:
                raise ValueError("No valid training examples provided")

            # Create SpaCy configuration
            config = {
                "lang": "en",
                "pipeline": ["ner"],
                "components": {
                    "ner": {
                        "factory": "ner",
                        "labels": list(self.ENTITY_TYPES.keys())
                    }
                },
                "training": {
                    "max_steps": iterations,
                    "dropout": dropout,
                    "batch_size": 16,
                    "optimizer": {
                        "@optimizers": "Adam.v1",
                        "learn_rate": 0.001
                    }
                }
            }

            # Save configuration
            config_path = Path(output_dir) / "config.cfg"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            spacy.util.save_config(config, config_path)

            # Train the model
            spacy.cli.train(
                str(config_path),
                output_path=output_dir,
                overrides={
                    "paths.train": None,
                    "paths.dev": None,
                    "training.max_steps": iterations
                },
                training_data=train_examples
            )

            logger.info(f"Model trained and saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, text: str) -> List[Dict[str, any]]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of extracted entities with their types and positions
        """
        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": self.ENTITY_TYPES.get(ent.label_, "")
                })

            return entities

        except Exception as e:
            logger.error(f"Error predicting entities: {str(e)}")
            raise

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            test_data: List of test examples

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Convert test data to SpaCy format
            test_examples = []
            for example in test_data:
                text = example["text"]
                entities = example["entities"]

                # Create Doc object
                doc = self.nlp.make_doc(text)

                # Add entity annotations
                ents = []
                for start, end, label in entities:
                    span = doc.char_span(start, end, label=label)
                    if span:
                        ents.append(span)
                    else:
                        logger.warning(f"Invalid span for entity: {text[start:end]}")

                if ents:  # Only add examples with valid entities
                    doc.ents = ents
                    test_examples.append(
                        Example.from_dict(doc, {"entities": [(e.start, e.end, e.label_) for e in doc.ents]}))

            if not test_examples:
                raise ValueError("No valid test examples provided")

            # Calculate metrics
            scores = {}
            for label in self.ENTITY_TYPES.keys():
                tp = 0  # True positives
                fp = 0  # False positives
                fn = 0  # False negatives

                for example in test_examples:
                    pred_ents = set((e.start, e.end, e.label_) for e in example.predicted.ents)
                    gold_ents = set((e.start, e.end, e.label_) for e in example.reference.ents)

                    # Count metrics for this label
                    label_pred = set(e for e in pred_ents if e[2] == label)
                    label_gold = set(e for e in gold_ents if e[2] == label)

                    tp += len(label_pred & label_gold)
                    fp += len(label_pred - label_gold)
                    fn += len(label_gold - label_pred)

                # Calculate precision, recall, and F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                scores[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

            # Calculate overall metrics
            overall = {
                "precision": sum(s["precision"] for s in scores.values()) / len(scores),
                "recall": sum(s["recall"] for s in scores.values()) / len(scores),
                "f1": sum(s["f1"] for s in scores.values()) / len(scores)
            }

            scores["overall"] = overall
            return scores

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
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

            self.nlp.to_disk(output_path)

            # Save entity type descriptions
            with open(output_path / "entity_types.json", "w") as f:
                json.dump(self.ENTITY_TYPES, f, indent=2)

            logger.info(f"Model saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, model_path: str) -> "LegalNERModel":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded LegalNERModel instance
        """
        try:
            model = cls(model_path)

            # Load entity type descriptions
            entity_types_path = Path(model_path) / "entity_types.json"
            if entity_types_path.exists():
                with open(entity_types_path) as f:
                    model.ENTITY_TYPES = json.load(f)

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise