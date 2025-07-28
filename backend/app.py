"""
Hugging Face Spaces entry point for AI Resume Reviewer
"""
import os
import logging
from app.main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (Remove all code in this file or delete the file if not needed for Gradio) 