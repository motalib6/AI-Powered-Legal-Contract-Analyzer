"""
Script to train all models for the Legal Contract Analyzer.
"""

import logging
from pathlib import Path
import json
import sys
import os
from typing import Dict, List, Tuple

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ner_model import LegalNERModel
from src.models.classifier import ClauseClassifier
from src.models.summarizer import ContractSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains all models for the Legal Contract Analyzer."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing prepared datasets
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load a prepared dataset.
        
        Args:
            dataset_name: Name of the dataset (ner, classifier, or summarization)
            
        Returns:
            Tuple of (training_data, test_data)
        """
        try:
            with open(self.data_dir / f"{dataset_name}_train.json") as f:
                train_data = json.load(f)
            
            with open(self.data_dir / f"{dataset_name}_test.json") as f:
                test_data = json.load(f)
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
            raise
    
    def train_ner_model(self):
        """Train the NER model."""
        try:
            logger.info("Training NER model...")
            
            # Load dataset
            train_data, test_data = self.load_dataset("ner")
            
            # Initialize model
            model = LegalNERModel()
            
            # Train model
            model.train(
                train_data=train_data,
                output_dir=str(self.models_dir / "ner"),
                iterations=30,
                dropout=0.2
            )
            
            # Evaluate model
            metrics = model.evaluate(test_data)
            logger.info(f"NER model evaluation metrics: {metrics}")
            
            # Save evaluation metrics
            with open(self.models_dir / "ner" / "evaluation_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("NER model training completed")
            
        except Exception as e:
            logger.error(f"Error training NER model: {str(e)}")
            raise
    
    def train_classifier(self):
        """Train the clause classifier."""
        try:
            logger.info("Training clause classifier...")
            
            # Load dataset
            train_data, test_data = self.load_dataset("classifier")
            
            # Initialize model
            model = ClauseClassifier(model_type="transformer")
            
            # Train model
            model.train(
                train_data=train_data,
                output_dir=str(self.models_dir / "classifier"),
                test_size=0.2,
                random_state=42
            )
            
            # Save evaluation metrics
            with open(self.models_dir / "classifier" / "evaluation_metrics.json", "r") as f:
                metrics = json.load(f)
            logger.info(f"Classifier evaluation metrics: {metrics}")
            
            logger.info("Clause classifier training completed")
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def train_summarizer(self):
        """Train the contract summarizer."""
        try:
            logger.info("Training contract summarizer...")
            
            # Load dataset
            train_data, test_data = self.load_dataset("summarization")
            
            # Initialize model
            model = ContractSummarizer()
            
            # Train model
            model.train(
                train_data=train_data,
                output_dir=str(self.models_dir / "summarizer"),
                num_epochs=3,
                batch_size=4,
                learning_rate=2e-5
            )
            
            # Evaluate model
            metrics = model.evaluate(test_data)
            logger.info(f"Summarizer evaluation metrics: {metrics}")
            
            # Save evaluation metrics
            with open(self.models_dir / "summarizer" / "evaluation_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Contract summarizer training completed")
            
        except Exception as e:
            logger.error(f"Error training summarizer: {str(e)}")
            raise

def main():
    """Main function to train all models."""
    try:
        logger.info("Starting model training...")
        
        trainer = ModelTrainer()
        
        # Train each model
        trainer.train_ner_model()
        trainer.train_classifier()
        trainer.train_summarizer()
        
        logger.info("All models trained successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 