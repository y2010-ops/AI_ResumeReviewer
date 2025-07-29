"use client";

import { useState } from "react";
import AnalysisResults from "@/components/AnalysisResults";
import ResumeUploader from "@/components/ResumeUploader";
import JobDescriptionInput from "@/components/JobDescriptionInput";
import LoadingSpinner from "@/components/LoadingSpinner";

interface AnalysisResult {
  final_similarity_score: number;
  final_similarity_percentage: number;
  similarity_category: string;
  skills_analysis: Record<string, unknown>;
  semantic_score: number;
  skills_score: number;
  llm_score: number;
  resume_bert_score: number;
  llm_details: Record<string, unknown>;
  confidence: number;
  anomaly: boolean;
  component_scores: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  debug_info: Record<string, unknown>;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalysis = async (formData: FormData) => {
    setError("");
    setResults(null);
    setLoading(true);
    try {
      // Use the environment variable directly, fallback to localhost:7860
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860';
      const endpoint = `${apiUrl}/api/v1/match`;
      
      console.log("Making request to:", endpoint);
      
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      
      // Log the raw response for debugging
      console.log("Raw response:", response);
      console.log("Response status:", response.status);
      console.log("Response headers:", response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        setError(`Analysis failed: ${response.status} ${errorText}`);
        setLoading(false);
        return;
      }
      
      let data;
      try {
        data = await response.json();
      } catch (parseError) {
        console.error("JSON parse error:", parseError);
        setError("Analysis failed: Invalid JSON from backend");
        setLoading(false);
        return;
      }
      
      // Log the parsed data for debugging
      console.log("Parsed data:", data);
      setResults(data);
    } catch (err) {
      console.error("Network error:", err);
      setError("Analysis failed: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJobDescription("");
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 relative overflow-hidden">
      {/* Header */}
      <header className="bg-slate-800/90 backdrop-blur-md sticky top-0 z-50 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform duration-300">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white via-blue-100 to-indigo-100 bg-clip-text text-transparent">
                  AI Resume Reviewer
                </h1>
                <p className="text-xs text-slate-300 font-medium">Powered by Advanced AI</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6">
              <div className="flex items-center space-x-3 text-slate-300 group">
                <div className="relative">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                </div>
                <span className="text-sm font-medium group-hover:text-white transition-colors duration-300">Real-time Analysis</span>
              </div>
              <div className="flex items-center space-x-3 text-slate-300 group">
                <div className="relative">
                  <div className="w-8 h-8 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                </div>
                <span className="text-sm font-medium group-hover:text-white transition-colors duration-300">AI-Powered</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative">
        {/* Hero Section */}
        <section className="relative py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12 animate-fade-in">
              <div className="relative overflow-hidden inline-flex items-center space-x-2 bg-gradient-to-b from-slate-800 to-slate-900 px-6 py-3 rounded-full mb-6 border-2 border-blue-400/60 shadow-[0_0_20px_rgba(59,130,246,0.3),0_0_40px_rgba(139,92,246,0.2)] transition-all duration-300">
                <span className="text-sm font-bold text-white relative z-10">AI-Powered Resume Analysis</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
                Get <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">AI-Analyzed</span> Resume Insights
              </h1>
              <p className="text-xl text-slate-300 max-w-4xl mx-auto leading-relaxed mb-8">
                Upload your resume and job description to receive comprehensive AI-powered analysis, 
                skill matching, and personalized recommendations to improve your job applications.
              </p>
              <div className="flex flex-wrap justify-center gap-4 text-sm text-slate-300">
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span>Instant Analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span>Skill Matching</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span>Personalized Insights</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Upload Section - Now contains both upload form and results */}
        <section className="relative py-16 bg-gradient-to-b from-slate-900 via-slate-900 to-black">
          <div className="max-w-5xl mx-auto px-6 py-8">
            <div className="bg-slate-800/90 backdrop-blur-xl rounded-3xl border-4 border-blue-200 p-8 md:p-12 relative shadow-[0_0_40px_rgba(59,130,246,0.9),0_0_80px_rgba(139,92,246,0.7),0_0_120px_rgba(59,130,246,0.5),0_0_20px_rgba(255,255,255,0.3)] transition-all duration-300">
              
              <div className="relative z-10">
                {results ? (
                  // Show Analysis Results inside the same box
                  <div>

                    {/* Results Header Inside Box */}
                    <div className="text-center mb-8">
                      <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 leading-tight tracking-tight">
                        <span className="bg-gradient-to-r from-blue-300 via-cyan-300 to-purple-300 bg-clip-text text-transparent">Analysis</span> Results
                      </h2>
                      <p className="text-lg text-slate-300 max-w-4xl mx-auto leading-relaxed mb-4 font-light">
                        Here&apos;s your personalized resume analysis with detailed insights and recommendations
                      </p>
                    </div>

                    {/* New Analysis Button */}
                    <div className="text-center mb-6">
                      <button
                        onClick={handleReset}
                        className="group relative inline-flex items-center justify-center px-4 py-2 text-xs font-semibold text-white bg-slate-800/90 backdrop-blur-xl rounded-full border-2 border-blue-400/50 shadow-[0_0_20px_rgba(59,130,246,0.4),0_0_40px_rgba(139,92,246,0.3)] transition-all duration-300 transform hover:scale-105"
                      >
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        New Analysis
                      </button>
                    </div>

                    {/* Analysis Results Content */}
                    <AnalysisResults results={results} onReset={handleReset} />
                  </div>
                ) : (
                  // Show Upload Form
                  <div className="text-center mb-12 mt-4">
                    <h2 className="text-3xl font-bold text-white mb-4">Upload Your Documents</h2>
                    <p className="text-slate-300">Get started with your AI-powered resume analysis</p>
                  </div>
                )}

                {!results && (
                  <div className="grid md:grid-cols-2 gap-8">
                    {/* Resume Upload */}
                    <div className="group">
                      <div className="flex items-center space-x-4 mb-6">
                        <div className="relative">
                          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                        </div>
                        <div>
                          <h3 className="text-xl font-semibold text-white">Upload Resume</h3>
                          <p className="text-sm text-slate-300">Upload your PDF resume</p>
                        </div>
                      </div>
                      <ResumeUploader onFileSelect={setFile} />
                    </div>

                    {/* Job Description */}
                    <div className="group">
                      <div className="flex items-center space-x-4 mb-6">
                        <div className="relative">
                          <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2-2v2m8 0V6a2 2 0 012 2v6a2 2 0 01-2 2H8a2 2 0 01-2-2V8a2 2 0 012-2h8z" />
                            </svg>
                          </div>
                        </div>
                        <div>
                          <h3 className="text-xl font-semibold text-white">Job Description</h3>
                          <p className="text-sm text-slate-300">Paste the job requirements</p>
                        </div>
                      </div>
                      <JobDescriptionInput value={jobDescription} onChange={setJobDescription} />
                    </div>
                  </div>
                )}

                {/* Analyze Button - Only show when not showing results */}
                {!results && (
                  <div className="text-center mt-10">
                    <button
                      onClick={() => {
                        if (file) {
                          const formData = new FormData();
                          formData.append('file', file);
                          formData.append('job_description', jobDescription);
                          handleAnalysis(formData);
                        }
                      }}
                      disabled={loading || !file || !jobDescription.trim()}
                      className="group relative inline-flex items-center justify-center px-12 py-6 text-xl font-bold text-white bg-gradient-to-r from-[#3b82f6] via-[#6366f1] to-[#8b5cf6] rounded-2xl hover:from-[#2563eb] hover:via-[#4f46e5] hover:to-[#7c3aed] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100 border border-blue-400/50"
                    >
                      <div className="relative flex items-center space-x-3">
                        {loading ? (
                          <LoadingSpinner />
                        ) : (
                          <>
                            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            <span>Analyze Resume</span>
                          </>
                        )}
                      </div>
                    </button>
                    <p className="text-sm text-slate-300 mt-4">Get instant AI-powered insights and recommendations</p>
                  </div>
                )}

                {error && (
                  <div className="mt-6 p-6 bg-red-900/50 border border-red-700/50 text-red-200 rounded-2xl max-w-2xl mx-auto animate-fade-in">
                    <div className="flex items-center space-x-3">
                      <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      <p className="text-sm font-medium">{error}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="relative py-20 bg-gradient-to-b from-black to-black">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">
                Advanced AI-Powered Analysis
              </h2>
              <p className="text-lg text-slate-300 max-w-3xl mx-auto">
                Our cutting-edge AI technology provides comprehensive resume analysis with multiple advanced models
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center group">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-4">LLM Intelligence</h3>
                <p className="text-slate-300 leading-relaxed">
                  Advanced language models analyze your resume content and provide intelligent insights about your qualifications and experience.
                </p>
              </div>

              <div className="text-center group">
                <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-4">Skill Extraction</h3>
                <p className="text-slate-300 leading-relaxed">
                  Advanced NLP techniques extract and match technical skills, soft skills, and domain-specific expertise from your resume.
                </p>
              </div>

              <div className="text-center group">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-4">Real-time Processing</h3>
                <p className="text-slate-300 leading-relaxed">
                  Lightning-fast analysis with real-time processing capabilities, providing instant feedback and recommendations.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Stack Section */}
        <section className="relative py-20 bg-black">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">
                Technology Stack
              </h2>
              <p className="text-lg text-slate-300 max-w-3xl mx-auto">
                Built with cutting-edge technologies for optimal performance and accuracy
              </p>
            </div>
            <div className="bg-slate-800/60 backdrop-blur-xl rounded-3xl shadow-2xl border border-slate-700/50 p-8">
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="text-center group">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">LLM Models</h3>
                  <p className="text-sm text-slate-300">Advanced language models for intelligent analysis</p>
                </div>

                <div className="text-center group">
                  <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">BERT Models</h3>
                  <p className="text-sm text-slate-300">Specialized models for semantic understanding</p>
                </div>

                <div className="text-center group">
                  <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Skill Extraction</h3>
                  <p className="text-sm text-slate-300">Advanced NLP for skill identification</p>
                </div>

                <div className="text-center group">
                  <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Real-time Processing</h3>
                  <p className="text-sm text-slate-300">Instant analysis and feedback</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Benefits Section */}
        <section className="relative py-20 bg-black">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">
                Why Choose Our Platform?
              </h2>
              <p className="text-lg text-slate-300 max-w-3xl mx-auto">
                Experience the advantages of our advanced AI-powered resume analysis platform
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="group bg-slate-800/60 backdrop-blur-xl rounded-3xl p-8 shadow-2xl hover:bg-slate-800/80 hover:shadow-3xl transition-all duration-500 transform hover:scale-105 border border-slate-700/50">
                <div className="flex items-center space-x-4 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-white">For Job Seekers</h3>
                </div>
                <ul className="space-y-4 text-slate-300">
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Identify skill gaps and improvement areas</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Get personalized recommendations</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Optimize resume for specific job requirements</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Increase interview chances with better matching</span>
                  </li>
                </ul>
              </div>

              <div className="group bg-slate-800/60 backdrop-blur-xl rounded-3xl p-8 shadow-2xl hover:bg-slate-800/80 hover:shadow-3xl transition-all duration-500 transform hover:scale-105 border border-slate-700/50">
                <div className="flex items-center space-x-4 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-white">Advanced Features</h3>
                </div>
                <ul className="space-y-4 text-slate-300">
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-emerald-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Multi-model AI analysis for accuracy</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-emerald-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Semantic similarity and skill matching</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-emerald-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Detailed breakdown of strengths and weaknesses</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <svg className="w-5 h-5 text-emerald-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>Actionable recommendations for improvement</span>
                  </li>
                </ul>
              </div>
        </div>
      </div>
        </section>
      </main>
    </div>
  );
} 
