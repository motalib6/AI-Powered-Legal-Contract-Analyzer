"""
Named Entity Recognition model for legal documents.
"""

import logging
from typing import List, Dict
import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

class LegalNERModel:
    """NER model for legal documents using spaCy."""
    
    def __init__(self):
        """Initialize the NER model."""
        self.model_loaded = False
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            # Add legal entity patterns
            ruler = self.nlp.add_pipe("entity_ruler")
            patterns = [
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "agreement"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "contract"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "clause"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "section"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "party"}, {"LOWER": "a"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "party"}, {"LOWER": "b"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "effective"}, {"LOWER": "date"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "termination"}, {"LOWER": "date"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "warranty"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "indemnification"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "liability"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "obligation"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "right"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "duty"}]},
                {"label": "LEGAL_TERM", "pattern": [{"LOWER": "responsibility"}]}
            ]
            ruler.add_patterns(patterns)
            self.model_loaded = True
            logger.info("NER model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NER model: {str(e)}")
            raise

    def predict(self, text: str) -> List[Dict]:
        """Predict entities in text."""
        if not self.model_loaded:
            raise RuntimeError("NER model not loaded")
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Calculate confidence based on entity length and type
                confidence = min(0.5 + (len(ent.text.split()) * 0.1), 0.9)
                
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "confidence": confidence,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error in NER prediction: {str(e)}")
            return [] 