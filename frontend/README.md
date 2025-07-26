# AI Resume-Job Matcher Frontend

A modern React/Next.js frontend for the AI-powered resume-job matching system.

## Features

- 📄 **Resume Upload**: Drag & drop PDF resume upload
- 📝 **Job Description Input**: Paste job descriptions for analysis
- 🤖 **AI Analysis**: Comprehensive matching using multiple AI models
- 📊 **Detailed Results**: Professional analysis with scores, skills, and recommendations
- 🎨 **Modern UI**: Clean, responsive design with Tailwind CSS

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Backend Connection

Make sure the backend server is running on `http://localhost:8000` before using the frontend.

## API Endpoints

The frontend connects to the following backend endpoints:

- `POST /api/v1/match` - Resume-job matching analysis

## Technologies Used

- **Next.js 15** - React framework
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## Project Structure

```
src/
├── app/
│   ├── page.tsx          # Main page component
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
└── components/
    ├── ResumeUploader.tsx    # PDF upload component
    ├── JobDescriptionInput.tsx # Job description input
    ├── AnalysisResults.tsx    # Results display
    └── LoadingSpinner.tsx     # Loading indicator
``` 