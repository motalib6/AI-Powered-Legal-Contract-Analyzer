"""
Script to prepare the CUAD dataset for training the Legal Contract Analyzer models.
"""

import logging
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple
import random
from sklearn.model_selection import train_test_split
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    """Prepares CUAD dataset for model training."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset preparator.
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_cuad(self):
        """Download CUAD dataset if not present."""
        try:
            # Check if dataset exists
            if not (self.data_dir / "cuad.json").exists():
                logger.info("Downloading CUAD dataset...")
                # TODO: Implement dataset download
                # For now, we'll assume the dataset is manually placed in the data directory
                raise FileNotFoundError(
                    "CUAD dataset not found. Please download it from "
                    "https://www.atticusprojectai.org/cuad and place it in the data directory."
                )
            else:
                logger.info("CUAD dataset found")
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def prepare_ner_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare data for NER model training.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        try:
            logger.info("Preparing NER training data...")
            
            # Load CUAD dataset
            with open(self.data_dir / "cuad.json") as f:
                cuad_data = json.load(f)
            
            # Convert to NER format
            ner_data = []
            for item in cuad_data["data"]:
                text = item["text"]
                entities = []
                
                # Extract entities from annotations
                for annotation in item.get("annotations", []):
                    if "entity" in annotation:
                        entity = annotation["entity"]
                        start = annotation["start"]
                        end = annotation["end"]
                        label = annotation["label"]
                        
                        entities.append((start, end, label))
                
                if entities:  # Only include examples with entities
                    ner_data.append({
                        "text": text,
                        "entities": entities
                    })
            
            # Split into train/test
            train_data, test_data = train_test_split(
                ner_data,
                test_size=0.2,
                random_state=42
            )
            
            # Save prepared data
            with open(self.data_dir / "ner_train.json", "w") as f:
                json.dump(train_data, f, indent=2)
            
            with open(self.data_dir / "ner_test.json", "w") as f:
                json.dump(test_data, f, indent=2)
            
            logger.info(f"Prepared {len(train_data)} training and {len(test_data)} test examples for NER")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing NER data: {str(e)}")
            raise
    
    def prepare_classifier_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare data for clause classifier training.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        try:
            logger.info("Preparing classifier training data...")
            
            # Load CUAD dataset
            with open(self.data_dir / "cuad.json") as f:
                cuad_data = json.load(f)
            
            # Convert to classifier format
            classifier_data = []
            for item in cuad_data["data"]:
                text = item["text"]
                
                # Extract clause type from annotations
                clause_type = None
                for annotation in item.get("annotations", []):
                    if "clause_type" in annotation:
                        clause_type = annotation["clause_type"]
                        break
                
                if clause_type:  # Only include examples with clause type
                    classifier_data.append({
                        "text": text,
                        "label": clause_type
                    })
            
            # Split into train/test
            train_data, test_data = train_test_split(
                classifier_data,
                test_size=0.2,
                random_state=42,
                stratify=[item["label"] for item in classifier_data]
            )
            
            # Save prepared data
            with open(self.data_dir / "classifier_train.json", "w") as f:
                json.dump(train_data, f, indent=2)
            
            with open(self.data_dir / "classifier_test.json", "w") as f:
                json.dump(test_data, f, indent=2)
            
            logger.info(f"Prepared {len(train_data)} training and {len(test_data)} test examples for classifier")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing classifier data: {str(e)}")
            raise
    
    def prepare_summarization_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare data for summarization model training.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        try:
            logger.info("Preparing summarization training data...")
            
            # Load CUAD dataset
            with open(self.data_dir / "cuad.json") as f:
                cuad_data = json.load(f)
            
            # Convert to summarization format
            summarization_data = []
            for item in cuad_data["data"]:
                text = item["text"]
                
                # Generate summary from annotations
                summary_points = []
                for annotation in item.get("annotations", []):
                    if "summary" in annotation:
                        summary_points.append(annotation["summary"])
                
                if summary_points:  # Only include examples with summaries
                    summary = " ".join(summary_points)
                    summarization_data.append({
                        "text": text,
                        "summary": summary
                    })
            
            # Split into train/test
            train_data, test_data = train_test_split(
                summarization_data,
                test_size=0.2,
                random_state=42
            )
            
            # Save prepared data
            with open(self.data_dir / "summarization_train.json", "w") as f:
                json.dump(train_data, f, indent=2)
            
            with open(self.data_dir / "summarization_test.json", "w") as f:
                json.dump(test_data, f, indent=2)
            
            logger.info(f"Prepared {len(train_data)} training and {len(test_data)} test examples for summarization")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing summarization data: {str(e)}")
            raise

def main():
    """Main function to prepare all training datasets."""
    try:
        logger.info("Starting dataset preparation...")
        
        preparator = DatasetPreparator()
        
        # Download dataset
        preparator.download_cuad()
        
        # Prepare data for each model
        preparator.prepare_ner_data()
        preparator.prepare_classifier_data()
        preparator.prepare_summarization_data()
        
        logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in dataset preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 