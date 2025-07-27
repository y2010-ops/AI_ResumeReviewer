# AI Resume Reviewer

An intelligent resume review and job matching system that uses AI to analyze resumes against job descriptions.

## 🚀 Quick Deploy

### Railway (Backend) + Vercel (Frontend)

**One-click deployment:**

1. **Backend (Railway):**
   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub repo
   - Set Root Directory: `backend`
   - Add environment variables
   - Deploy

2. **Frontend (Vercel):**
   - Go to [Vercel.com](https://vercel.com)
   - Connect your GitHub repo
   - Set Root Directory: `frontend`
   - Add `NEXT_PUBLIC_API_URL` environment variable
   - Deploy

📖 **Detailed deployment guide:** [DEPLOYMENT.md](./DEPLOYMENT.md)

## ✨ Features

- **PDF Resume Processing**: Extract and analyze text from PDF resumes
- **AI-Powered Analysis**: Uses multiple LLM APIs (Groq, Together AI, Cohere) for comprehensive analysis
- **Semantic Similarity**: BERT-based semantic matching between resume and job description
- **Skills Extraction**: Advanced skills matching with importance weighting
- **Multi-Layer Validation**: Ensures accurate scoring with confidence metrics
- **Modern Web Interface**: Built with Next.js and Tailwind CSS
- **RESTful API**: FastAPI backend with comprehensive endpoints

## 🏗️ Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Supabase**: Database for storing review history
- **Machine Learning**: 
  - Sentence Transformers for semantic similarity
  - spaCy for NLP and skills extraction
  - Multiple LLM APIs for intelligent analysis
- **PDF Processing**: pdfplumber and PyMuPDF
- **Deployment**: Railway (recommended) or Vercel

### Frontend
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling
- **Deployment**: Vercel

## 📁 Project Structure

```
AI_ResumeReviewer/
├── backend/                 # Railway deployment
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── config.py       # Configuration settings
│   │   ├── embedding.py    # AI analysis engine
│   │   ├── supabase.py     # Database operations
│   │   └── routes/
│   │       └── review.py   # API endpoints
│   ├── requirements.txt    # Python dependencies
│   ├── railway.json       # Railway configuration
│   └── env.example        # Environment variables template
├── frontend/               # Vercel deployment
│   ├── src/
│   │   ├── app/           # Next.js App Router
│   │   └── components/    # React components
│   ├── package.json       # Node.js dependencies
│   ├── vercel.json        # Vercel configuration
│   └── next.config.ts     # Next.js configuration
├── DEPLOYMENT.md          # Detailed deployment guide
├── setup-local.bat        # Windows local setup
├── setup-local.sh         # Unix local setup
└── README.md             # This file
```

## 🚀 Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Quick Start

**Windows:**
```bash
# Run the setup script
setup-local.bat
```

**Unix/Linux/macOS:**
```bash
# Make script executable
chmod +x setup-local.sh

# Run the setup script
./setup-local.sh
```

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🔧 Environment Variables

### Backend (Railway)
```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
TOGETHER_API_KEY=your_together_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Frontend (Vercel)
```env
NEXT_PUBLIC_API_URL=https://your-railway-backend-url.railway.app
```

## 📊 API Endpoints

### POST `/api/v1/match`
Analyze resume against job description

**Request:**
- `file`: PDF resume file
- `job_description`: Job description text

**Response:**
```json
{
  "final_similarity_score": 0.85,
  "final_similarity_percentage": 85.0,
  "similarity_category": "Excellent Match (85-100%)",
  "skills_analysis": {
    "coverage_percentage": 0.75,
    "direct_match_count": 15,
    "total_job_skills": 20,
    "missing_skills": ["Docker", "Kubernetes"]
  },
  "llm_details": {
    "strengths": ["Strong technical background", "Relevant experience"],
    "gaps": ["Missing cloud experience"],
    "recommendations": ["Consider AWS certification"]
  }
}
```

### GET `/health`
Health check endpoint

### GET `/api/v1/reviews`
Get recent resume reviews (if Supabase configured)

## 🎯 Features in Detail

### AI Analysis Engine
- **Multi-LLM Ensemble**: Uses Groq, Together AI, and Cohere APIs
- **Fallback Strategy**: Automatically switches between APIs for reliability
- **Semantic Analysis**: BERT-based similarity scoring
- **Skills Matching**: Fuzzy matching with importance weighting

### PDF Processing
- **Multiple Extractors**: pdfplumber and PyMuPDF for reliability
- **Text Cleaning**: Advanced text preprocessing
- **Error Handling**: Graceful fallbacks for corrupted files

### Skills Analysis
- **Dynamic Database**: 100+ technical skills with fuzzy matching
- **Importance Weighting**: High/medium/low importance levels
- **Related Skills**: Compensation for missing skills with related ones
- **Coverage Analysis**: Detailed skills gap analysis

### Validation System
- **Multi-Layer Scoring**: Combines semantic, skills, and LLM scores
- **Confidence Metrics**: Reliability indicators
- **Anomaly Detection**: Flags inconsistent results
- **Score Adjustment**: Intelligent score normalization

## 🔒 Security

- **CORS Configuration**: Properly configured for production
- **File Validation**: PDF type and size validation
- **Environment Variables**: Secure API key management
- **Input Sanitization**: Protection against malicious inputs

## 📈 Performance

- **Model Caching**: Efficient model loading and caching
- **Async Processing**: Non-blocking API calls
- **Memory Optimization**: Efficient text processing
- **Response Time**: Typically 5-15 seconds for analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Deployment Issues**: Check [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Local Development**: Use the setup scripts provided
- **API Documentation**: Available at `/docs` when backend is running
- **GitHub Issues**: Create an issue for bugs or feature requests

---

**Made with ❤️ for better job matching** 