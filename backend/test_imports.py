#!/usr/bin/env python3
"""
Test script to check imports in HF Spaces environment
"""
import sys
import os

print("=== Import Test for HF Spaces ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test basic imports
try:
    import gradio
    print("✅ gradio imported successfully")
except ImportError as e:
    print(f"❌ gradio import failed: {e}")

try:
    import sentence_transformers
    print("✅ sentence_transformers imported successfully")
except ImportError as e:
    print(f"❌ sentence_transformers import failed: {e}")

try:
    import transformers
    print("✅ transformers imported successfully")
except ImportError as e:
    print(f"❌ transformers import failed: {e}")

try:
    import torch
    print("✅ torch imported successfully")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

# Test app imports
try:
    from app.embedding import CorrectedResumeJobMatcher
    print("✅ CorrectedResumeJobMatcher imported successfully")
except ImportError as e:
    print(f"❌ CorrectedResumeJobMatcher import failed: {e}")

print("=== Import Test Complete ===") 