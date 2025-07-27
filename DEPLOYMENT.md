# 🚀 Deployment Guide: Railway (Backend) + Vercel (Frontend)

This guide will walk you through deploying the AI Resume Reviewer with **Railway for backend** and **Vercel for frontend**.

## 📋 Prerequisites

- GitHub account
- Railway account (free tier available)
- Vercel account (free tier available)
- Git installed locally

## 🔧 Step 1: Prepare Your Project

### 1.1 Verify Project Structure

Your project should have this structure:
```
AI_ResumeReviewer/
├── backend/           # Railway deployment
│   ├── app/
│   ├── requirements.txt
│   ├── railway.json   # Railway config
│   └── env.example
├── frontend/          # Vercel deployment
│   ├── src/
│   ├── package.json
│   ├── vercel.json    # Vercel config
│   └── next.config.ts
└── README.md
```

### 1.2 Environment Variables Setup

**Backend Environment Variables (Railway):**
```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

**Frontend Environment Variables (Vercel):**
```env
NEXT_PUBLIC_API_URL=https://your-railway-backend-url.railway.app
```

## 🚂 Step 2: Deploy Backend to Railway

### 2.1 Create Railway Account
1. Go to [Railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Complete the setup

### 2.2 Deploy Backend
1. **Click "New Project"**
2. **Select "Deploy from GitHub repo"**
3. **Choose your repository**: `y2010-ops/AI_ResumeReviewer`
4. **Set Root Directory**: `backend`
5. **Click "Deploy"**

### 2.3 Configure Railway Settings
1. **Go to your project** in Railway dashboard
2. **Click "Settings"**
3. **Add Environment Variables**:
   ```
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   TOGETHER_API_KEY=your_together_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

### 2.4 Railway Configuration
The `backend/railway.json` file is already configured:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### 2.5 Get Backend URL
1. **Wait for deployment to complete**
2. **Copy your Railway URL** (e.g., `https://your-app.railway.app`)
3. **Test the health endpoint**: `https://your-app.railway.app/health`

## 🎨 Step 3: Deploy Frontend to Vercel

### 3.1 Create Vercel Account
1. Go to [Vercel.com](https://vercel.com)
2. Sign up with your GitHub account

### 3.2 Deploy Frontend
1. **Go to Vercel Dashboard**
2. **Click "New Project"**
3. **Import your GitHub repository**: `y2010-ops/AI_ResumeReviewer`
4. **Configure settings**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

### 3.3 Set Environment Variables
In Vercel project settings, add:
```
NEXT_PUBLIC_API_URL=https://your-railway-backend-url.railway.app
```

### 3.4 Deploy
1. **Click "Deploy"**
2. **Wait for deployment to complete**
3. **Your frontend will be available** at the provided URL

## 🔄 Step 4: Test Your Deployment

### 4.1 Test Backend (Railway)
```bash
# Health check
curl https://your-railway-backend-url.railway.app/health

# Expected response:
# {"status": "healthy"}
```

### 4.2 Test Frontend (Vercel)
1. **Visit your Vercel URL**
2. **Upload a sample PDF resume**
3. **Enter a job description**
4. **Verify the analysis works**

### 4.3 Test API Integration
```bash
# Test the match endpoint
curl -X POST "https://your-railway-backend-url.railway.app/api/v1/match" \
  -F "file=@sample_resume.pdf" \
  -F "job_description=We are looking for a Python developer..."
```

## 🛠️ Step 5: Troubleshooting

### 5.1 Railway Issues

**Build Failures:**
- Check `backend/requirements.txt` for correct dependencies
- Verify Python version (Railway uses Python 3.11)
- Check logs in Railway dashboard

**Runtime Errors:**
- Verify environment variables are set
- Check the health endpoint: `/health`
- Review Railway logs

**Common Railway Commands:**
```bash
# View logs
railway logs

# Check status
railway status

# Restart service
railway restart
```

### 5.2 Vercel Issues

**Build Failures:**
- Check `frontend/package.json` for correct dependencies
- Verify Node.js version (Vercel uses Node.js 18+)
- Check build logs in Vercel dashboard

**Runtime Errors:**
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check browser console for errors
- Review Vercel function logs

### 5.3 CORS Issues
If you get CORS errors, the backend is already configured to allow all origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Step 6: Monitoring

### 6.1 Railway Monitoring
- **Logs**: View in Railway dashboard
- **Metrics**: CPU, memory usage
- **Deployments**: Automatic on Git push

### 6.2 Vercel Monitoring
- **Analytics**: Built-in Vercel Analytics
- **Functions**: View function logs
- **Performance**: Core Web Vitals

## 🔐 Step 7: Security

### 7.1 Environment Variables
- **Never commit API keys** to Git
- **Use Railway/Vercel environment variables**
- **Rotate keys regularly**

### 7.2 API Security
- **Consider rate limiting** for production
- **Add authentication** if needed
- **Monitor API usage**

## 🎉 Success!

Your AI Resume Reviewer is now deployed:

**URLs:**
- **Frontend**: `https://your-frontend.vercel.app`
- **Backend**: `https://your-backend.railway.app`
- **API Docs**: `https://your-backend.railway.app/docs`

## 📞 Support

**Railway Support:**
- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)

**Vercel Support:**
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Community](https://github.com/vercel/vercel/discussions)

---

**Happy Deploying! 🚀** 