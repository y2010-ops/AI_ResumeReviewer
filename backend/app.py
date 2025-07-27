"""
Hugging Face Spaces entry point for AI Resume Reviewer
"""
import os
import gradio as gr
import logging
import re
import io
import pdfplumber
import fitz  # PyMuPDF
import requests
from typing import Dict, Any
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Resume Reviewer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pdfplumber and PyMuPDF as fallback"""
    # Try pdfplumber first
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            if text.strip():
                return clean_text(text)
    except Exception:
        pass
    
    # Fallback to PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return clean_text(text)
    except Exception:
        pass
    
    return ""

def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def extract_skills(text: str) -> list:
    """Extract skills from text using simple keyword matching"""
    skills_db = [
        "python", "java", "javascript", "react", "angular", "vue", "node.js",
        "sql", "mysql", "postgresql", "mongodb", "aws", "azure", "gcp",
        "docker", "kubernetes", "git", "html", "css", "api", "rest",
        "machine learning", "ai", "data science", "pandas", "numpy",
        "tensorflow", "pytorch", "django", "flask", "spring", "express"
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)
    
    return found_skills

def simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def analyze_resume_api(resume_bytes: bytes, job_description: str) -> Dict[str, Any]:
    """API version of resume analysis that returns JSON"""
    try:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(resume_bytes)
        if not resume_text:
            return {"error": "Could not extract text from PDF"}
        
        # Extract skills
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        
        # Calculate similarity scores
        semantic_score = simple_similarity(resume_text, job_description)
        skills_score = simple_similarity(" ".join(resume_skills), " ".join(job_skills))
        
        # Calculate overall score
        final_score = (semantic_score * 0.6 + skills_score * 0.4) * 100
        
        # Determine category
        if final_score >= 80:
            category = "Excellent Match (80-100%)"
        elif final_score >= 60:
            category = "Good Match (60-79%)"
        elif final_score >= 40:
            category = "Fair Match (40-59%)"
        else:
            category = "Poor Match (<40%)"
        
        # Find missing skills
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]
        
        # Create response
        return {
            "final_similarity_percentage": final_score,
            "similarity_category": category,
            "semantic_score": semantic_score,
            "skills_score": skills_score,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "missing_skills": missing_skills,
            "skills_analysis": {
                "resume_skills_count": len(resume_skills),
                "job_skills_count": len(job_skills),
                "matched_skills_count": len(set(resume_skills).intersection(set(job_skills))),
                "coverage_percentage": len(set(resume_skills).intersection(set(job_skills))) / len(job_skills) if job_skills else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {"error": str(e)}

def analyze_resume(resume_file, job_description):
    """Gradio version of resume analysis that returns formatted text"""
    try:
        if resume_file is None:
            return "Please upload a PDF resume file."
        
        if not job_description or len(job_description.strip()) < 50:
            return "Please provide a detailed job description (at least 50 characters)."
        
        # Read file content
        with open(resume_file.name, 'rb') as f:
            resume_bytes = f.read()
        
        # Get analysis results
        result = analyze_resume_api(resume_bytes, job_description)
        
        if "error" in result:
            return f"❌ **Error**: {result['error']}"
        
        # Format output for Gradio
        final_score = result["final_similarity_percentage"]
        category = result["similarity_category"]
        semantic_score = result["semantic_score"]
        skills_score = result["skills_score"]
        resume_skills = result["resume_skills"]
        missing_skills = result["missing_skills"]
        skills_analysis = result["skills_analysis"]
        
        output = f"""
## 🎯 **Match Analysis Results**

### **Overall Score: {final_score:.1f}%**
**Category:** {category}

### **Skills Analysis**
- **Resume Skills Found:** {skills_analysis['resume_skills_count']}
- **Job Skills Required:** {skills_analysis['job_skills_count']}
- **Skills Coverage:** {skills_analysis['matched_skills_count']}/{skills_analysis['job_skills_count']} matched

### **Component Scores**
- **Semantic Similarity:** {semantic_score*100:.1f}%
- **Skills Matching:** {skills_score*100:.1f}%

### **Resume Skills**
{', '.join(resume_skills[:10]) if resume_skills else 'None identified'}

### **Missing Skills**
{', '.join(missing_skills[:10]) if missing_skills else 'None identified'}

### **Recommendations**
"""
        
        # Add recommendations based on score
        if final_score >= 80:
            output += "✅ Strong match! Your resume aligns well with the job requirements."
        elif final_score >= 60:
            output += "⚠️ Good match with room for improvement. Consider highlighting relevant experience."
        elif final_score >= 40:
            output += "📝 Fair match. Focus on acquiring missing skills and improving relevant experience."
        else:
            output += "❌ Poor match. Consider applying to roles that better align with your skills."
        
        return output
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"❌ **Analysis failed:** {str(e)}\n\nPlease check your input and try again."

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "AI Resume Reviewer API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Resume Reviewer API"}

@app.post("/api/v1/match")
async def match_resume_job(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """API endpoint for resume-job matching"""
    try:
        # Read file content
        resume_bytes = await file.read()
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return {"error": "Only PDF files are supported"}
        
        # Validate job description
        if len(job_description.strip()) < 50:
            return {"error": "Job description must be at least 50 characters long"}
        
        # Perform analysis
        result = analyze_resume_api(resume_bytes, job_description)
        
        return result
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return {"error": str(e)}

# Create Gradio interface
interface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"]),
        gr.Textbox(
            label="📝 Job Description", 
            placeholder="Enter the job description here...",
            lines=8
        )
    ],
    outputs=gr.Markdown(label="📊 Analysis Results"),
    title="🤖 AI Resume Reviewer",
    description="""
    **Advanced AI-powered resume review and job matching system**
    
    Upload your PDF resume and provide a job description to get a comprehensive analysis including:
    - Overall match score
    - Skills analysis
    - Component breakdown
    - Recommendations
    """,
    examples=[
        [
            None,
            "Software Engineer position requiring Python, JavaScript, React, and 3+ years of experience in web development. Responsibilities include building scalable applications, collaborating with cross-functional teams, and mentoring junior developers."
        ]
    ],
    theme=gr.themes.Soft()
)

# Launch the interface
if __name__ == "__main__":
    interface.launch() 