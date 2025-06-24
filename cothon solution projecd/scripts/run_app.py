"""
Script to run the Legal Contract Analyzer Streamlit application.
"""

import logging
import sys
import os
from pathlib import Path
import json
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.main import ContractAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_streamlit():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="SK MOTALIB Legal Contract Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .upload-section {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .result-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Run the Streamlit application."""
    try:
        logger.info("Starting Legal Contract Analyzer application...")
        
        # Setup Streamlit
        setup_streamlit()
        
        # Initialize analyzer
        analyzer = ContractAnalyzer()
        
        # Title and description
        st.title("SK MOTALIB Legal Contract Analyzer")
        st.markdown("""
            This application uses AI to analyze legal contracts, extract key clauses,
            classify risks, and generate summaries. Upload your contract document
            (PDF or DOCX) to get started.
        """)
        
        # File upload section
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Contract Document",
                type=["pdf", "docx"],
                help="Upload a PDF or DOCX file containing a legal contract"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Analyze contract
                with st.spinner("Analyzing contract..."):
                    results = analyzer.analyze_contract(str(temp_path))
                
                # Display results
                with st.container():
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    
                    # Summary
                    st.subheader("Contract Summary")
                    st.write(results["summary"])
                    
                    # Key Clauses
                    st.subheader("Key Clauses")
                    for clause in results["clauses"]:
                        with st.expander(f"{clause['type']} (Confidence: {clause['confidence']:.2f})"):
                            st.write(clause["text"])
                    
                    # Risk Analysis
                    st.subheader("Risk Analysis")
                    risk_df = pd.DataFrame(results["risks"])
                    st.dataframe(risk_df)
                    
                    # Entities
                    st.subheader("Key Entities")
                    for entity_type, entities in results["entities"].items():
                        with st.expander(entity_type):
                            for entity in entities:
                                st.write(f"- {entity['text']} (Confidence: {entity['confidence']:.2f})")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temporary file
                temp_path.unlink()
                
            except Exception as e:
                logger.error(f"Error analyzing contract: {str(e)}")
                st.error(f"Error analyzing contract: {str(e)}")
        
        # Sidebar
        with st.sidebar:
            st.header("About")
            st.markdown("""
                This application uses advanced AI models to analyze legal contracts:
                
                - **Named Entity Recognition**: Identifies key entities like parties,
                  dates, and amounts
                - **Clause Classification**: Categorizes contract clauses
                - **Risk Analysis**: Identifies potential risks and obligations
                - **Contract Summarization**: Generates concise summaries
                
                The models are trained on the CUAD dataset and fine-tuned for legal
                contract analysis.
            """)
            
            st.header("Model Information")
            st.markdown("""
                - NER Model: SpaCy-based custom model
                - Classifier: Legal-BERT fine-tuned model
                - Summarizer: BART-based model
            """)
            
            if st.button("View Model Metrics"):
                try:
                    # Load and display model metrics
                    metrics_dir = Path("models")
                    
                    for model_name in ["ner", "classifier", "summarizer"]:
                        metrics_path = metrics_dir / model_name / "evaluation_metrics.json"
                        if metrics_path.exists():
                            with open(metrics_path) as f:
                                metrics = json.load(f)
                            st.subheader(f"{model_name.upper()} Model Metrics")
                            st.json(metrics)
                
                except Exception as e:
                    logger.error(f"Error loading model metrics: {str(e)}")
                    st.error("Error loading model metrics")
        
        logger.info("Application running successfully")
        
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        st.error("An error occurred while running the application")
        raise

if __name__ == "__main__":
    main() 