"""
Hugging Face Spaces entry point for AI Resume Reviewer
"""
import os
import gradio as gr
import logging
from app.main import app
from app.embedding import CorrectedResumeJobMatcher
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_resume(resume_file, job_description):
    """Analyze resume with job description using the full ML backend"""
    try:
        if resume_file is None:
            return "Please upload a PDF resume file."
        
        if not job_description or len(job_description.strip()) < 50:
            return "Please provide a detailed job description (at least 50 characters)."
        
        # Read file content
        with open(resume_file.name, 'rb') as f:
            resume_bytes = f.read()
        
        # Initialize the full ML matcher
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        
        matcher = CorrectedResumeJobMatcher(
            groq_api_key=groq_api_key,
            cohere_api_key=cohere_api_key,
            resume_bert_model=None  # Auto-select best model
        )
        
        # Perform the full analysis
        result = matcher.match(resume_bytes, job_description)
        
        # Format output for Gradio
        final_score = result["final_similarity_percentage"]
        category = result["similarity_category"]
        confidence = result["confidence"]
        
        # Skills analysis
        skills_analysis = result.get("skills_analysis", {})
        coverage = skills_analysis.get("coverage_percentage", 0) * 100
        missing_skills = skills_analysis.get("missing_skills", [])
        
        # LLM details
        llm_details = result.get("llm_details", {})
        api_used = llm_details.get("api_used", "N/A")
        
        # Format output
        output = f"""
## 🎯 **Match Analysis Results**

### **Overall Score: {final_score:.1f}%**
**Category:** {category}
**Confidence:** {confidence:.1%}

### **Skills Analysis**
- **Coverage:** {coverage:.1f}%
- **Missing Skills:** {len(missing_skills)}
- **API Used:** {api_used}

### **Component Scores**
- **Semantic Similarity:** {result['semantic_score']*100:.1f}%
- **Skills Matching:** {result['skills_score']*100:.1f}%
- **Enhanced Skills:** {result['enhanced_skills_score']*100:.1f}%
- **Resume-BERT:** {result['resume_bert_score']*100:.1f}%
- **LLM Assessment:** {result['llm_score']:.1f}/100

### **Missing Skills**
{', '.join(missing_skills[:10]) if missing_skills else 'None identified'}

### **Recommendations**
"""
        
        # Add LLM recommendations if available
        if llm_details.get("recommendations"):
            for i, rec in enumerate(llm_details["recommendations"][:5], 1):
                output += f"{i}. {rec}\n"
        else:
            output += "Focus on acquiring missing skills and improving relevant experience."
        
        return output
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"❌ **Analysis failed:** {str(e)}\n\nPlease check your input and try again."

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