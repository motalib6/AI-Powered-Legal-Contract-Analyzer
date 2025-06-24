"""
Contract summarization module using NLTK and scikit-learn.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractSummarizer:
    """Contract summarization using extractive methods."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the summarizer.
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.stop_words = set(stopwords.words('english'))
        
        # Add legal-specific stop words
        legal_stop_words = {
            'hereby', 'herein', 'hereof', 'hereto', 'hereunder',
            'thereby', 'therein', 'thereof', 'thereto', 'thereunder',
            'whereby', 'wherein', 'whereof', 'whereto', 'whereunder',
            'party', 'parties', 'agreement', 'contract', 'clause',
            'section', 'article', 'paragraph', 'subparagraph'
        }
        self.stop_words.update(legal_stop_words)
        
        logger.info("Initialized summarizer with TF-IDF")
    
    def summarize(self, text: str, max_length: int = 150) -> Dict:
        """
        Summarize the input text using extractive methods.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in sentences
            
        Returns:
            Dictionary containing summary and metrics
        """
        try:
            if not text.strip():
                return {
                    "summary": "",
                    "original_length": 0,
                    "summary_length": 0,
                    "compression_ratio": 0
                }
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            sentences = sent_tokenize(cleaned_text)
            original_length = len(sentences)
            
            if original_length <= max_length:
                return {
                    "summary": cleaned_text,
                    "original_length": original_length,
                    "summary_length": original_length,
                    "compression_ratio": 1.0
                }
            
            # Get sentence scores using TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-max_length:][::-1]
            top_indices.sort()  # Sort by position in document
            
            # Extract summary
            summary_sentences = [sentences[i] for i in top_indices]
            summary = " ".join(summary_sentences)
            
            # Calculate metrics
            summary_length = len(summary_sentences)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return {
                "summary": "",
                "original_length": len(sent_tokenize(text)),
                "summary_length": 0,
                "compression_ratio": 0
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^\w\s.,;:!?()]', ' ', text)
            
            # Fix spacing around punctuation
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)
            
            # Remove multiple periods
            text = re.sub(r'\.+', '.', text)
            
            # Ensure proper sentence endings
            text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text."""
        try:
            # Tokenize and clean words
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in self.stop_words and w.isalnum()]
            
            # Get word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top phrases
            phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [phrase for phrase, _ in phrases[:max_phrases]]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []
    
    @classmethod
    def load(cls, model_path: str) -> "ContractSummarizer":
        """Load a pre-trained summarizer."""
        return cls()
    
    def save(self, model_path: str):
        """Save the summarizer model."""
        try:
            # Save vectorizer
            import joblib
            joblib.dump(self.vectorizer, Path(model_path) / "vectorizer.joblib")
            logger.info(f"Saved summarizer to {model_path}")
        except Exception as e:
            logger.error(f"Error saving summarizer: {str(e)}")
            raise