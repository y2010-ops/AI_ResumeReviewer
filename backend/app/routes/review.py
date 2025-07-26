"""
FastAPI routes for resume review functionality
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.embedding import CorrectedResumeJobMatcher
from app.supabase import supabase_service
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/match", response_model=dict)
async def match_resume_job(
    file: UploadFile = File(..., description="PDF resume file"),
    job_description: str = Form(..., description="Job description text", min_length=50)
):
    """
    Enhanced resume-job matching using the CorrectedResumeJobMatcher
    
    Args:
        file: PDF file upload
        job_description: Job description as form data
        
    Returns:
        dict: Comprehensive matching results with detailed analysis
    """
    try:
        # Validate file type
        if file.content_type not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only PDF files are allowed."
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Initialize the enhanced matcher
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        
        matcher = CorrectedResumeJobMatcher(
            groq_api_key=groq_api_key,
            cohere_api_key=cohere_api_key,
            resume_bert_model=None  # Auto-select best model
        )
        
        # Perform the matching analysis
        result = matcher.match(file_content, job_description)
        
        # Print detailed analysis to terminal
        print("\n" + "="*80)
        print("🔍 API REQUEST ANALYSIS RESULTS")
        print("="*80)
        
        final_score = result["final_similarity_score"]
        final_percentage = result["final_similarity_percentage"]
        category = result["similarity_category"]
        
        print(f"\n🎯 FINAL MATCH SCORE: {final_score:.4f} ({final_percentage:.2f}%)")
        print(f"📊 CATEGORY: {category}")
        print(f"🔍 CONFIDENCE: {result['confidence']:.3f}")
        print(f"⚠️  ANOMALY: {result['anomaly']}")
        
        # Component scores
        print(f"\n📊 COMPONENT SCORES:")
        print(f"   • Semantic Similarity: {result['semantic_score']:.3f} ({result['semantic_score']*100:.1f}%)")
        print(f"   • Skills Matching: {result['skills_score']:.3f} ({result['skills_score']*100:.1f}%)")
        print(f"   • Enhanced Skills: {result['enhanced_skills_score']:.3f} ({result['enhanced_skills_score']*100:.1f}%)")
        print(f"   • Resume-BERT: {result['resume_bert_score']:.3f} ({result['resume_bert_score']*100:.1f}%)")
        print(f"   • LLM Assessment: {result['llm_score']:.1f}/100")
        
        # LLM Details
        if result.get('llm_details'):
            llm_details = result['llm_details']
            print(f"\n🧠 LLM ANALYSIS:")
            print(f"   • API Used: {llm_details.get('api_used', 'N/A')}")
            print(f"   • Response Time: {llm_details.get('response_time', 0):.2f}s")
            print(f"   • Compatibility: {llm_details.get('compatibility_score', 0)}/100")
            
            if llm_details.get('strengths'):
                print(f"   • Key Strengths: {len(llm_details['strengths'])} identified")
            if llm_details.get('gaps'):
                print(f"   • Areas for Improvement: {len(llm_details['gaps'])} identified")
            if llm_details.get('recommendations'):
                print(f"   • Recommendations: {len(llm_details['recommendations'])} provided")
        
        # Skills Analysis
        if result.get('skills_analysis'):
            skills_analysis = result['skills_analysis']
            print(f"\n🔧 SKILLS ANALYSIS:")
            print(f"   • Coverage: {skills_analysis['coverage_percentage']*100:.1f}%")
            print(f"   • Direct Matches: {skills_analysis['direct_match_count']}/{skills_analysis['total_job_skills']}")
            print(f"   • Missing Skills: {len(skills_analysis['missing_skills'])}")
            print(f"   • Critical Skills Missing: {len(skills_analysis['critical_skills_missing'])}")
        
        # Model Info
        if result.get('model_info'):
            model_info = result['model_info']
            print(f"\n🤖 MODEL INFO:")
            print(f"   • Primary Model: {model_info.get('primary_semantic_model', 'N/A')}")
            print(f"   • Resume Model: {model_info.get('resume_specific_model', 'N/A')}")
            print(f"   • Total Models: {model_info.get('total_models_loaded', 0)}")
        
        print("\n" + "="*80)
        
        # Store results in database if Supabase is configured
        try:
            # Extract resume text for storage
            resume_text = matcher.pdf_extractor.extract_text(file_content)
            
            # Create a summary for storage
            feedback = f"Match Score: {result['final_similarity_percentage']:.1f}% - {result['similarity_category']}"
            
            # Store in database
            await supabase_service.insert_resume_review(
                resume_text=resume_text,
                job_description=job_description,
                match_score=result['final_similarity_percentage'],
                feedback=feedback
            )
        except Exception as e:
            logger.warning(f"Failed to store results in database: {str(e)}")
            # Continue without failing the request
        
        return result
        
    except Exception as e:
        logger.error(f"Resume matching failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process resume matching: {str(e)}"
        )

@router.get("/reviews")
async def get_recent_reviews():
    """
    Get recent resume reviews
    
    Returns:
        list: Recent resume reviews
    """
    try:
        reviews = await supabase_service.get_resume_reviews(limit=10)
        return {"reviews": reviews}
    except Exception as e:
        logger.error(f"Error retrieving reviews: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve reviews"
        )

@router.get("/health")
async def health_check():
    """
    Health check for the review service
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_model": settings.LLM_MODEL_NAME,
        "supabase_connected": supabase_service.client is not None
    } 