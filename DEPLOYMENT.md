# 🚀 Deployment Guide

This guide will walk you through deploying the AI Resume Reviewer to GitHub and Vercel.

## 📋 Prerequisites

- GitHub account
- Vercel account
- Git installed locally
- Node.js 18+ and Python 3.11+ (for local testing)

## 🔧 Step 1: Prepare Your Project

### 1.1 Initialize Git Repository

```bash
# Initialize git in your project directory
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: AI Resume Reviewer"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/ai-resume-reviewer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 1.2 Environment Variables Setup

Create the following environment files:

**Backend (.env)**
```env
# API Keys (Optional - for enhanced analysis)
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
TOGETHER_API_KEY=your_together_api_key_here

# Database (Optional - for storing results)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

**Frontend (.env.local)**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🌐 Step 2: Deploy Backend to Vercel

### 2.1 Create Vercel Backend Project

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Configure the following settings:

**Build Settings:**
- Framework Preset: Other
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
- Output Directory: Leave empty
- Install Command: Leave empty

**Environment Variables:**
Add these in the Vercel dashboard:
```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
TOGETHER_API_KEY=your_together_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 2.2 Deploy Backend

1. Click "Deploy"
2. Wait for deployment to complete
3. Note your backend URL (e.g., `https://your-backend.vercel.app`)

## 🎨 Step 3: Deploy Frontend to Vercel

### 3.1 Create Vercel Frontend Project

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import the same GitHub repository
4. Configure the following settings:

**Build Settings:**
- Framework Preset: Next.js
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `.next`
- Install Command: `npm install`

**Environment Variables:**
```
NEXT_PUBLIC_API_URL=https://your-backend-url.vercel.app
```

### 3.2 Deploy Frontend

1. Click "Deploy"
2. Wait for deployment to complete
3. Your frontend will be available at the provided URL

## 🔄 Step 4: Set Up GitHub Actions (Optional)

### 4.1 Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:
```
VERCEL_TOKEN=your_vercel_token
VERCEL_ORG_ID=your_vercel_org_id
VERCEL_BACKEND_PROJECT_ID=your_backend_project_id
VERCEL_FRONTEND_PROJECT_ID=your_frontend_project_id
```

### 4.2 Get Vercel Tokens

1. Go to [Vercel Account Settings](https://vercel.com/account/tokens)
2. Create a new token
3. Copy the token to `VERCEL_TOKEN`

### 4.3 Get Project IDs

1. Go to your Vercel project settings
2. Copy the Project ID from the General tab
3. Use this for the respective project ID secrets

## 🐳 Step 5: Docker Deployment (Alternative)

### 5.1 Local Docker Testing

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### 5.2 Production Docker Deployment

```bash
# Build backend image
cd backend
docker build -t ai-resume-reviewer-backend .

# Build frontend image
cd ../frontend
docker build -t ai-resume-reviewer-frontend .

# Run containers
docker run -d -p 8000:8000 --name backend ai-resume-reviewer-backend
docker run -d -p 3000:3000 --name frontend ai-resume-reviewer-frontend
```

## 🔧 Step 6: Alternative Backend Deployment

### 6.1 Railway Deployment

1. Go to [Railway](https://railway.app)
2. Connect your GitHub repository
3. Set environment variables
4. Deploy automatically

### 6.2 Render Deployment

1. Go to [Render](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## 🧪 Step 7: Testing Your Deployment

### 7.1 Health Check

```bash
# Test backend health
curl https://your-backend-url.vercel.app/health

# Expected response:
# {"status": "healthy"}
```

### 7.2 API Testing

```bash
# Test the match endpoint
curl -X POST "https://your-backend-url.vercel.app/api/v1/match" \
  -F "file=@sample_resume.pdf" \
  -F "job_description=We are looking for a Python developer..."
```

### 7.3 Frontend Testing

1. Visit your frontend URL
2. Upload a sample PDF resume
3. Enter a job description
4. Verify the analysis works correctly

## 🔍 Step 8: Monitoring and Debugging

### 8.1 Vercel Logs

1. Go to your Vercel project dashboard
2. Click on "Functions" tab
3. View logs for any errors

### 8.2 Environment Variable Issues

Common issues:
- Missing API keys
- Incorrect URLs
- CORS configuration

### 8.3 Performance Optimization

1. **Backend:**
   - Enable Vercel Edge Functions
   - Use caching for model loading
   - Optimize model sizes

2. **Frontend:**
   - Enable Next.js Image Optimization
   - Use CDN for static assets
   - Implement proper caching

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Python/Node.js versions
   - Verify all dependencies are installed
   - Check for syntax errors

2. **API Errors**
   - Verify environment variables
   - Check CORS configuration
   - Ensure backend is running

3. **Model Loading Issues**
   - Check internet connectivity
   - Verify model download paths
   - Monitor memory usage

### Debug Commands

```bash
# Check backend logs
vercel logs your-backend-project

# Check frontend logs
vercel logs your-frontend-project

# Test API locally
curl -X POST "http://localhost:8000/api/v1/match" \
  -F "file=@test.pdf" \
  -F "job_description=test"
```

## 📊 Step 9: Analytics and Monitoring

### 9.1 Vercel Analytics

1. Enable Vercel Analytics in your project
2. Monitor performance metrics
3. Track user interactions

### 9.2 Custom Monitoring

Consider adding:
- Error tracking (Sentry)
- Performance monitoring
- User analytics

## 🔐 Step 10: Security Considerations

### 10.1 Environment Variables

- Never commit API keys to Git
- Use Vercel's environment variable encryption
- Rotate keys regularly

### 10.2 CORS Configuration

Update your backend CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🎉 Success!

Your AI Resume Reviewer is now deployed and ready to use!

**Your URLs:**
- Frontend: `https://your-frontend.vercel.app`
- Backend: `https://your-backend.vercel.app`
- API Docs: `https://your-backend.vercel.app/docs`

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review Vercel logs
3. Create an issue on GitHub
4. Contact support with specific error messages

---

**Happy Deploying! 🚀** 