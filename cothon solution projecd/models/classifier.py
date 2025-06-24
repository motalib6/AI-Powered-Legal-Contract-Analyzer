"""
Clause classification model for legal documents.
"""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

class ClauseClassifier:
    """Classifier for legal clauses using rule-based approach."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.model_loaded = False
        try:
            # Define clause patterns
            self.patterns = {
                "DEFINITION": [
                    r"means\s+.*",
                    r"shall\s+mean\s+.*",
                    r"defined\s+as\s+.*"
                ],
                "OBLIGATION": [
                    r"shall\s+.*",
                    r"must\s+.*",
                    r"agrees\s+to\s+.*",
                    r"will\s+.*"
                ],
                "RESTRICTION": [
                    r"shall\s+not\s+.*",
                    r"may\s+not\s+.*",
                    r"cannot\s+.*",
                    r"prohibited\s+from\s+.*"
                ],
                "TERMINATION": [
                    r"terminat(e|ion)\s+.*",
                    r"end\s+of\s+.*",
                    r"expir(e|ation)\s+.*"
                ],
                "LIABILITY": [
                    r"liability\s+.*",
                    r"damages\s+.*",
                    r"indemnif(y|ication)\s+.*",
                    r"warrant(y|ies)\s+.*"
                ],
                "CONFIDENTIALITY": [
                    r"confidential\s+.*",
                    r"secret\s+.*",
                    r"non-disclosure\s+.*",
                    r"proprietary\s+.*"
                ]
            }
            
            # Compile patterns
            self.compiled_patterns = {
                category: [re.compile(pattern, re.IGNORECASE) 
                          for pattern in patterns]
                for category, patterns in self.patterns.items()
            }
            
            self.model_loaded = True
            logger.info("Clause classifier initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing clause classifier: {str(e)}")
            raise

    def predict(self, text: str) -> Dict:
        """Classify a clause."""
        if not self.model_loaded:
            raise RuntimeError("Classifier not loaded")
        
        try:
            # Initialize scores
            scores = {category: 0.0 for category in self.patterns.keys()}
            
            # Check each pattern
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        # Increase score based on pattern match
                        scores[category] += 0.3
                        # Additional score for multiple matches
                        matches = len(pattern.findall(text))
                        if matches > 1:
                            scores[category] += min(0.2 * (matches - 1), 0.4)
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v/total_score for k, v in scores.items()}
            
            # Get best category
            best_category = max(scores.items(), key=lambda x: x[1])
            
            return {
                "category": best_category[0],
                "confidence": best_category[1],
                "scores": scores
            }
        except Exception as e:
            logger.error(f"Error in clause classification: {str(e)}")
            return {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "scores": {}
            } 