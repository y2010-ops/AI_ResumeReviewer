---
title: AI Resume Reviewer Backend
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# AI Resume Reviewer - Backend API

Advanced AI-powered resume review and job matching system optimized for Hugging Face Spaces deployment.

## 🚀 Features

- **PDF Resume Processing**: Extract and analyze text from PDF resumes
- **AI-Powered Analysis**: Multiple LLM APIs (Groq, Together AI, Cohere)
- **Semantic Similarity**: BERT-based semantic matching
- **Skills Extraction**: Advanced skills matching with importance weighting
- **Multi-Layer Validation**: Accurate scoring with confidence metrics
- **GPU Acceleration**: Optimized for HF Spaces GPU deployment

## 🏗️ Tech Stack

- **FastAPI**: Modern Python web framework
- **Machine Learning**: 
  - Sentence Transformers for semantic similarity
  - spaCy for NLP and skills extraction
  - Multiple LLM APIs for intelligent analysis
- **PDF Processing**: pdfplumber and PyMuPDF
- **Deployment**: Hugging Face Spaces (GPU optimized)

## 🔧 API Endpoints

- `GET /` - Health check
- `GET /health` - Health check endpoint
- `POST /api/v1/match` - Resume-job matching
- `GET /api/v1/reviews` - Recent reviews
- `GET /docs` - Interactive API documentation

## 📝 Usage

1. Upload a PDF resume
2. Provide job description
3. Get comprehensive analysis with:
   - Overall match score
   - Component analysis
   - Skills assessment
   - Recommendations

## 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
TOGETHER_API_KEY=your_together_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## 🚀 HF Spaces Deployment

This backend is optimized for Hugging Face Spaces deployment:

- ✅ **GPU Support**: Automatic GPU acceleration for ML models
- ✅ **Auto Model Loading**: spaCy models downloaded automatically
- ✅ **Memory Optimized**: Efficient ML model loading
- ✅ **FastAPI Integration**: Native FastAPI support
- ✅ **CORS Configured**: Ready for frontend integration

## 📊 Performance

- **CPU Mode**: Suitable for development and testing
- **GPU Mode**: Recommended for production (faster inference)
- **Memory**: Optimized for HF Spaces memory limits
- **Startup Time**: ~30-60 seconds (model loading)

---

**Ready for HF Spaces deployment! 🚀** 