# AI Resume Reviewer

An intelligent resume review and job matching system that uses AI to analyze resumes against job descriptions.

## Features

- **PDF Resume Processing**: Extract and analyze text from PDF resumes
- **AI-Powered Analysis**: Uses multiple LLM APIs (Groq, Together AI, Cohere) for comprehensive analysis
- **Semantic Similarity**: BERT-based semantic matching between resume and job description
- **Skills Extraction**: Advanced skills matching with importance weighting
- **Multi-Layer Validation**: Ensures accurate scoring with confidence metrics
- **Modern Web Interface**: Built with Next.js and Tailwind CSS
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Supabase**: Database for storing review history
- **Multiple LLM APIs**: Groq, Together AI, Cohere for AI analysis
- **Sentence Transformers**: BERT-based semantic similarity
- **spaCy**: Natural language processing for skills extraction
- **PyMuPDF & pdfplumber**: PDF text extraction

### Frontend
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client for API calls

## Project Structure

```
AI_ResumeReviewer/
├── backend/
│   ├── app/
│   │   ├── config.py          # Configuration settings
│   │   ├── embedding.py       # AI analysis engine
│   │   ├── main.py           # FastAPI application
│   │   ├── supabase.py       # Database operations
│   │   └── routes/
│   │       └── review.py     # API endpoints
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile           # Backend container
│   └── vercel.json          # Vercel deployment config
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js app directory
│   │   └── components/      # React components
│   ├── package.json         # Node.js dependencies
│   ├── Dockerfile           # Frontend container
│   └── vercel.json          # Vercel deployment config
├── .github/                 # GitHub Actions CI/CD
├── docker-compose.yml       # Local development setup
└── README.md               # This file
```

## GitHub Upload & Vercel Deployment

### ✅ Files Removed for Clean Repository
- `parsed_resume.json` - Temporary error file
- `.cursorignore` - Empty IDE file
- `frontend/.git/` - Nested git repository
- `backend/package-lock.json` - Empty file (not needed for Python)
- `frontend/package-lock.json` - Will be regenerated on deployment

### ✅ Updated .gitignore
The `.gitignore` file has been enhanced to exclude:
- Environment files (`.env*`)
- Dependencies (`node_modules/`, `__pycache__/`)
- Build outputs (`.next/`, `dist/`)
- Lock files (will be regenerated)
- Temporary files
- IDE-specific files

### 🚀 Ready for Deployment
The project is now optimized for:
- **GitHub**: Clean repository without unnecessary files
- **Vercel**: Both frontend and backend have proper `vercel.json` configurations
- **CI/CD**: GitHub Actions workflow for automated testing and deployment

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AI_ResumeReviewer
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Environment Configuration**
   
   Create `.env` file in `backend/`:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```
   
   Create `.env.local` file in `frontend/`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

5. **Start Development Servers**
   ```bash
   # Backend (from backend directory)
   uvicorn app.main:app --reload
   
   # Frontend (from frontend directory)
   npm run dev
   ```

### Vercel Deployment

1. **Connect to Vercel**
   - Push your code to GitHub
   - Connect your repository to Vercel
   - Vercel will automatically detect the Next.js frontend

2. **Backend Deployment**
   - Vercel will deploy the backend from the `backend/` directory
   - Set environment variables in Vercel dashboard

3. **Environment Variables**
   Set these in your Vercel project settings:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```

## API Endpoints

### POST `/api/v1/match`
Analyze resume against job description.

**Request:**
- `file`: PDF resume file
- `job_description`: Job description text

**Response:**
```json
{
  "final_similarity_score": 0.85,
  "final_similarity_percentage": 85.0,
  "similarity_category": "Excellent Match (85-100%)",
  "skills_analysis": {...},
  "llm_details": {...},
  "confidence": 0.92
}
```

### GET `/api/v1/reviews`
Get recent review history.

### GET `/health`
Health check endpoint.

## Features in Detail

### AI Analysis Engine
- **Multi-API Ensemble**: Uses Groq, Together AI, and Cohere for robust analysis
- **Fallback Strategy**: Automatically switches between APIs for reliability
- **Smart Prompting**: Chain-of-thought prompts for comprehensive analysis
- **Response Validation**: Ensures consistent JSON output format

### Semantic Matching
- **BERT Models**: Uses multiple specialized BERT models for different domains
- **Resume-Specific Models**: Optimized for resume/job matching
- **Ensemble Scoring**: Combines multiple models for accuracy
- **Domain Analysis**: Detects technical, business, or legal contexts

### Skills Extraction
- **NER-based Extraction**: Uses spaCy for named entity recognition
- **Fuzzy Matching**: Handles variations in skill names
- **Importance Weighting**: Prioritizes critical skills
- **Related Skills**: Identifies transferable skills

### Multi-Layer Validation
- **Score Consistency**: Detects and corrects scoring anomalies
- **Confidence Metrics**: Provides reliability indicators
- **Component Analysis**: Detailed breakdown of each scoring component
- **Diagnostic Information**: Helps understand scoring decisions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team. 