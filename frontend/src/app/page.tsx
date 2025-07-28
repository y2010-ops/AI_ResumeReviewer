/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import * as React from 'react';
import { useState } from 'react';
import ResumeUploader from '@/components/ResumeUploader';
import JobDescriptionInput from '@/components/JobDescriptionInput';
import AnalysisResults from '@/components/AnalysisResults';
import LoadingSpinner from '@/components/LoadingSpinner';

// Define apiUrl at the top level so process.env is replaced at build time
const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://yaminivj-ai-resumereviewer.hf.space/';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState('');
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalysis = async (formData: FormData) => {
    setLoading(true);
    setError(null);
    
    try {
      // Debug: Log the FormData contents
      console.log('FormData contents:');
      for (const [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }
      
      console.log('Sending request to:', `${apiUrl}/api/v1/match`);
      
      const response = await fetch(`${apiUrl}/api/v1/match`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type or Content-Length - let browser handle it for FormData
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      console.log('Success response:', data);
      setResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJobDescription('');
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            AI Resume Reviewer
          </h1>
          <p className="text-lg text-gray-600">
            Upload your resume and job description to get AI-powered analysis
          </p>
        </div>

        {!results ? (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <ResumeUploader onFileSelect={setFile} />
            </div>
            
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <JobDescriptionInput 
                value={jobDescription}
                onChange={setJobDescription}
              />
            </div>

            {file && jobDescription && (
              <div className="text-center">
                <button
                  onClick={() => {
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('job_description', jobDescription);
                    handleAnalysis(formData);
                  }}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-bold py-3 px-8 rounded-lg text-lg transition-colors"
                >
                  {loading ? <LoadingSpinner /> : 'Analyze Resume'}
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
                {error}
              </div>
            )}
          </div>
        ) : (
          <AnalysisResults results={results} onReset={handleReset} />
        )}
      </div>
    </div>
  );
} 