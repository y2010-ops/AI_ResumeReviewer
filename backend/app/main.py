"""
FastAPI main application entry point for AI Resume Reviewer
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from app.routes import review
from app.config import settings

# Create FastAPI app instance
app = FastAPI(
    title="AI Resume Reviewer API",
    description="Backend API for AI-powered resume review and job matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(review.router, prefix="/api/v1", tags=["review"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Resume Reviewer API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 