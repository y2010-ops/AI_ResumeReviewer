'use client';

import { useState, useEffect } from 'react';
import ResumeUploader from '@/components/ResumeUploader';
import JobDescriptionInput from '@/components/JobDescriptionInput';
import AnalysisResults from '@/components/AnalysisResults';
import LoadingSpinner from '@/components/LoadingSpinner';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleAnalysis = async (formData: FormData) => {
    setIsLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/match', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze resume');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isClient) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner />
          <p className="text-gray-600 mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            AI Resume-Job Matcher
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload your resume and paste a job description to get a comprehensive 
            analysis of your fit for the position using advanced AI technology.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {!results && !isLoading && (
            <form id="analysis-form" className="bg-white rounded-lg shadow-lg p-6 mb-8">
              <ResumeUploader />
              <JobDescriptionInput />
              <button
                type="button"
                onClick={() => {
                  const form = document.getElementById('analysis-form') as HTMLFormElement;
                  if (form) {
                    const formData = new FormData(form);
                    handleAnalysis(formData);
                  }
                }}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 mt-6"
              >
                Analyze Resume-Job Match
              </button>
            </form>
          )}

          {isLoading && (
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
              <LoadingSpinner />
              <p className="text-gray-600 mt-4">
                Analyzing your resume against the job description...
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-8">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {results && (
            <AnalysisResults results={results} onReset={() => setResults(null)} />
          )}
        </div>
      </div>
    </div>
  );
} 