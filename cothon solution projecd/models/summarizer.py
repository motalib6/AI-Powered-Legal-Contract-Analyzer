"""
Contract summarization model.
"""

import logging
from typing import Dict, List
from nltk.tokenize import sent_tokenize
import re

logger = logging.getLogger(__name__)

class ContractSummarizer:
    """Summarizer for legal contracts using extractive summarization."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.model_loaded = False
        try:
            # Define important legal terms and their weights
            self.legal_terms = {
                "agreement": 2.0,
                "contract": 2.0,
                "party": 1.5,
                "shall": 1.5,
                "obligation": 1.5,
                "right": 1.5,
                "duty": 1.5,
                "liability": 1.5,
                "warranty": 1.5,
                "indemnification": 1.5,
                "termination": 1.5,
                "confidential": 1.5,
                "effective": 1.0,
                "date": 1.0,
                "term": 1.0,
                "condition": 1.0
            }
            
            # Compile patterns for section headers
            self.section_patterns = [
                re.compile(r"^[A-Z][A-Za-z\s]+:$"),
                re.compile(r"^\d+\.\s+[A-Z][A-Za-z\s]+$"),
                re.compile(r"^[A-Z][A-Za-z\s]+\s+[A-Z][A-Za-z\s]+:$")
            ]
            
            self.model_loaded = True
            logger.info("Contract summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing summarizer: {str(e)}")
            raise

    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header."""
        return any(pattern.match(text.strip()) for pattern in self.section_patterns)

    def _calculate_sentence_score(self, sentence: str) -> float:
        """Calculate importance score for a sentence."""
        # Base score
        score = 1.0
        
        # Check for legal terms
        sentence_lower = sentence.lower()
        for term, weight in self.legal_terms.items():
            if term in sentence_lower:
                score += weight
        
        # Penalize very long sentences
        words = sentence.split()
        if len(words) > 30:
            score *= 0.8
        
        # Boost section headers
        if self._is_section_header(sentence):
            score *= 1.5
        
        return score

    def summarize(self, text: str, max_sentences: int = 5) -> Dict:
        """Generate a summary of the contract."""
        if not self.model_loaded:
            raise RuntimeError("Summarizer not loaded")
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Calculate scores for each sentence
            sentence_scores = [
                (sentence, self._calculate_sentence_score(sentence))
                for sentence in sentences
            ]
            
            # Sort sentences by score
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top sentences
            summary_sentences = [s[0] for s in sentence_scores[:max_sentences]]
            
            # Sort summary sentences by original position
            summary_sentences.sort(key=lambda x: sentences.index(x))
            
            # Calculate statistics
            original_length = len(text.split())
            summary_length = len(" ".join(summary_sentences).split())
            
            return {
                "summary": " ".join(summary_sentences),
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": summary_length / original_length if original_length > 0 else 0,
                "key_terms": list(self.legal_terms.keys())
            }
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": "Error generating summary. Please check the document format.",
                "original_length": len(text.split()),
                "summary_length": 0,
                "compression_ratio": 0,
                "key_terms": []
            } 