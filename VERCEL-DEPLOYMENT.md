# 🚀 Vercel Frontend Deployment Guide

Complete guide to deploy your AI Resume Reviewer frontend on Vercel.

## 📋 Prerequisites

- GitHub repository: `y2010-ops/AI_ResumeReviewer`
- Vercel account: [vercel.com](https://vercel.com)
- Backend already deployed on HF Spaces

## 🎯 Step 1: Prepare Your HF Spaces Backend URL

### 1.1 Get Your Backend URL
Your HF Spaces backend should be running at:
```
https://your-username-ai-resume-reviewer-backend.hf.space
```

### 1.2 Test Your Backend
Visit your HF Spaces URL and verify:
- ✅ Gradio interface loads
- ✅ PDF upload works
- ✅ Analysis completes successfully

## 🎨 Step 2: Deploy Frontend to Vercel

### 2.1 Create Vercel Account
1. Go to [Vercel.com](https://vercel.com)
2. Sign up with your GitHub account
3. Complete the setup

### 2.2 Import Repository
1. **Click "New Project"**
2. **Import Git Repository**
3. **Select**: `y2010-ops/AI_ResumeReviewer`
4. **Click "Import"**

### 2.3 Configure Project Settings
1. **Framework Preset**: Next.js (should auto-detect)
2. **Root Directory**: `frontend`
3. **Build Command**: `npm run build`
4. **Output Directory**: `.next`
5. **Install Command**: `npm install`

### 2.4 Add Environment Variable
**Add this environment variable:**
```
NEXT_PUBLIC_API_URL=https://your-username-ai-resume-reviewer-backend.hf.space
```

**Steps:**
1. In Vercel project settings, go to "Environment Variables"
2. Add new variable:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your HF Spaces URL
   - **Environment**: Production, Preview, Development
3. Click "Save"

### 2.5 Deploy
1. **Click "Deploy"**
2. **Wait for build completion** (2-5 minutes)
3. **Get your frontend URL** (e.g., `https://your-project.vercel.app`)

## 🔧 Configuration Files

### Frontend Files (Ready for Vercel)
- ✅ `package.json` - Dependencies and scripts
- ✅ `next.config.ts` - Next.js configuration
- ✅ `vercel.json` - Vercel configuration
- ✅ `tsconfig.json` - TypeScript configuration
- ✅ `src/app/page.tsx` - Main page component
- ✅ `src/components/` - React components

### Vercel Configuration
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "installCommand": "npm install"
}
```

## 🧪 Testing Your Deployment

### 2.1 Frontend Health Check
- Visit your Vercel URL
- Should show the AI Resume Reviewer interface
- No error messages

### 2.2 Integration Test
1. **Upload a PDF resume**
2. **Enter job description**
3. **Click "Analyze Resume"**
4. **Should connect to HF Spaces backend**
5. **Get analysis results**

### 2.3 Expected Flow
```
Frontend (Vercel) → Backend (HF Spaces) → Analysis Results
```

## 🔑 Environment Variables Guide

### Required
```env
NEXT_PUBLIC_API_URL=https://your-username-ai-resume-reviewer-backend.hf.space
```

### Optional (for development)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🐛 Common Issues & Solutions

### ❌ Error: "API URL not found"
**Solution**: Check environment variable in Vercel settings

### ❌ Error: "CORS error"
**Solution**: Backend has CORS configured, should work

### ❌ Error: "Build failed"
**Solution**: Check build logs in Vercel dashboard

### ❌ Error: "API timeout"
**Solution**: HF Spaces may be slow, increase timeout

## ✅ Success Indicators

- [ ] Vercel deployment completes
- [ ] Frontend loads without errors
- [ ] PDF upload works
- [ ] Job description input works
- [ ] Analysis button responds
- [ ] Results display correctly
- [ ] Backend integration works

## 🚀 Final URLs

### Your Deployment URLs:
- **Frontend**: `https://your-project.vercel.app`
- **Backend**: `https://your-username-ai-resume-reviewer-backend.hf.space`

### Test Your Full Application:
1. Visit your Vercel frontend URL
2. Upload a resume and job description
3. Get AI-powered analysis results

## 📊 Performance

- **Frontend**: Vercel CDN (fast global delivery)
- **Backend**: HF Spaces (CPU/GPU processing)
- **Integration**: REST API calls between services

## 🔄 Updates

### To Update Frontend:
1. Push changes to GitHub
2. Vercel auto-deploys
3. No manual intervention needed

### To Update Backend:
1. Push changes to GitHub
2. HF Spaces auto-rebuilds
3. Frontend continues working

---

**✅ Ready for Vercel deployment! 🚀** 