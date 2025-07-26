"""
Supabase client setup and database operations
"""
from supabase import create_client, Client
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for Supabase database operations"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.client: Optional[Client] = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup Supabase client"""
        try:
            if settings.SUPABASE_URL and settings.SUPABASE_KEY:
                self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not provided - database operations will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self.client = None
    
    async def insert_resume_review(
        self, 
        resume_text: str, 
        job_description: str, 
        match_score: float, 
        feedback: str
    ) -> Optional[Dict[str, Any]]:
        """
        Insert resume review into database
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description
            match_score: Similarity score
            feedback: AI-generated feedback
            
        Returns:
            Dict with inserted record or None if failed
        """
        if not self.client:
            logger.warning("Supabase client not available - skipping database insert")
            return None
            
        try:
            data = {
                "resume_text": resume_text,
                "job_description": job_description,
                "match_score": match_score,
                "feedback": feedback,
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.table("resume_reviews").insert(data).execute()
            
            if result.data:
                logger.info("Resume review stored in database successfully")
                return result.data[0]
            else:
                logger.error("Failed to insert resume review - no data returned")
                return None
                
        except Exception as e:
            logger.error(f"Database insert failed: {str(e)}")
            return None
    
    async def get_resume_reviews(self, limit: int = 10) -> list:
        """
        Get recent resume reviews
        
        Args:
            limit: Maximum number of reviews to return
            
        Returns:
            List of recent reviews
        """
        if not self.client:
            logger.warning("Supabase client not available - returning empty list")
            return []
            
        try:
            result = self.client.table("resume_reviews").select("*").order("created_at", desc=True).limit(limit).execute()
            
            if result.data:
                logger.info(f"Retrieved {len(result.data)} recent reviews")
                return result.data
            else:
                logger.info("No recent reviews found")
                return []
                
        except Exception as e:
            logger.error(f"Failed to retrieve reviews: {str(e)}")
            return []

# Global service instance
supabase_service = SupabaseService() 