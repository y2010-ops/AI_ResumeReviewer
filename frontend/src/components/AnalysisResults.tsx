/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState } from "react";

interface AnalysisResultsProps {
  results: any;
  onReset: () => void;
}

export default function AnalysisResults({ results }: AnalysisResultsProps) {
  const [expandedMatchedSkills, setExpandedMatchedSkills] = useState(false);
  const [expandedMissingSkills, setExpandedMissingSkills] = useState(false);
  
  const formatScore = (value: number) => `${(value * 100).toFixed(1)}%`;

  const getRecommendation = (score: number) => {
    if (score >= 0.8) return "Your resume shows strong alignment with the job requirements.";
    if (score >= 0.6) return "Your resume shows good alignment with the job requirements.";
    if (score >= 0.4) return "Your resume shows fair alignment with the job requirements.";
    return "Your resume needs improvement to match the job requirements.";
  };

  // Extract backend data with fallbacks
  const llmDetails = results.llm_details || {};
  const skillsAnalysis = results.skills_analysis || {};
  const skillsRecommendations = results.skills_recommendations || [];
  
  // LLM Analysis Data
  const strengths = llmDetails.strengths || [];
  const gaps = llmDetails.gaps || [];
  const recommendations = llmDetails.recommendations || [];
  
  // Skills Analysis Data
  const directMatches = skillsAnalysis.direct_matches || [];
  const missingSkills = skillsAnalysis.missing_skills || [];
  
  // Analysis Quality
  const confidence = results.confidence || 0;
  const anomaly = results.anomaly || false;

  return (
    <div className="space-y-6">
      {/* Three Main Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Card 1: Overall Match */}
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/50 shadow-lg">
          <div className="text-center">
            {/* Circular Progress */}
            <div className="relative inline-block mb-4">
              <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="44"
                  fill="none"
                  stroke="#475569"
                  strokeWidth="6"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="44"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="6"
                  strokeDasharray={`${2 * Math.PI * 44}`}
                  strokeDashoffset={`${2 * Math.PI * 44 * (1 - results.final_similarity_score)}`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-2xl font-bold text-emerald-400 leading-none">
                  {formatScore(results.final_similarity_score)}
                </div>
                <div className="text-xs text-slate-400 mt-1">Match Score</div>
              </div>
            </div>
            
            <h3 className="text-lg font-bold text-white mb-2">Overall Match</h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              {getRecommendation(results.final_similarity_score)}
            </p>
            <div className="mt-3 text-xs text-slate-400">
              <div>Confidence: {(confidence * 100).toFixed(1)}%</div>
              <div className={anomaly ? "text-red-400" : "text-emerald-400"}>
                {anomaly ? "⚠️ Anomaly Detected" : "✅ Consistent Analysis"}
              </div>
            </div>
          </div>
        </div>

        {/* Card 2: Matched Skills */}
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/50 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-emerald-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <h3 className="text-lg font-bold text-white">Matched Skills</h3>
            </div>
            {directMatches.length > 6 && (
              <button
                onClick={() => setExpandedMatchedSkills(!expandedMatchedSkills)}
                className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
              >
                {expandedMatchedSkills ? "Show Less" : `Show All (${directMatches.length})`}
              </button>
            )}
          </div>
          
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {directMatches.length > 0 ? (
              <>
                {directMatches.slice(0, expandedMatchedSkills ? directMatches.length : 6).map((skill: string, index: number) => (
                  <div key={index} className="flex items-center justify-between bg-emerald-500/10 rounded-lg px-3 py-2 border border-emerald-500/30">
                    <span className="text-sm font-medium text-emerald-300">{skill}</span>
                    <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                ))}
                {!expandedMatchedSkills && directMatches.length > 6 && (
                  <div className="text-center py-2">
                    <span className="text-xs text-slate-400">+{directMatches.length - 6} more skills</span>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-4">
                <span className="text-sm text-slate-400">No matched skills found</span>
              </div>
            )}
          </div>
        </div>

        {/* Card 3: Missing Skills */}
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/50 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <h3 className="text-lg font-bold text-white">Missing Skills</h3>
            </div>
            {missingSkills.length > 6 && (
              <button
                onClick={() => setExpandedMissingSkills(!expandedMissingSkills)}
                className="text-xs text-red-400 hover:text-red-300 transition-colors"
              >
                {expandedMissingSkills ? "Show Less" : `Show All (${missingSkills.length})`}
              </button>
            )}
          </div>
          
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {missingSkills.length > 0 ? (
              <>
                {missingSkills.slice(0, expandedMissingSkills ? missingSkills.length : 6).map((skill: string, index: number) => (
                  <div key={index} className="flex items-center justify-between bg-red-500/10 rounded-lg px-3 py-2 border border-red-500/30">
                    <span className="text-sm font-medium text-red-300">{skill}</span>
                    <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                ))}
                {!expandedMissingSkills && missingSkills.length > 6 && (
                  <div className="text-center py-2">
                    <span className="text-xs text-slate-400">+{missingSkills.length - 6} more skills</span>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-4">
                <span className="text-sm text-slate-400">No missing skills identified</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Skills Analysis */}
      <div className="bg-slate-700/30 rounded-xl p-6 border border-slate-600/30">
        <h3 className="text-lg font-semibold text-white mb-4">Skills Analysis</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-slate-300">Skills Coverage</span>
            <span className="text-slate-400 font-medium">
              {skillsAnalysis.coverage_percentage ? `${(skillsAnalysis.coverage_percentage * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-300">Direct Matches</span>
            <span className="text-slate-400 font-medium">
              {skillsAnalysis.direct_match_count || 0}/{skillsAnalysis.total_job_skills || 0}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-300">Missing Skills</span>
            <span className="text-slate-400 font-medium">
              {missingSkills.length}
            </span>
          </div>
        </div>
      </div>

      {/* Consolidated Analysis Card */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/50 shadow-lg">
        <div className="space-y-6">
          {/* Key Strengths Section */}
          <div>
            <h3 className="text-xl font-bold text-white mb-4">Key Strengths</h3>
            {strengths.length > 0 ? (
              <ul className="space-y-3 text-slate-300">
                {strengths.map((strength: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-emerald-400 mr-3 text-lg">•</span>
                    <span>{strength}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-400 italic">No specific strengths identified in the analysis.</p>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-slate-600/50"></div>

          {/* Areas for Improvement Section */}
          <div>
            <h3 className="text-xl font-bold text-white mb-4">Areas for Improvement</h3>
            {gaps.length > 0 ? (
              <ul className="space-y-3 text-slate-300">
                {gaps.map((gap: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-amber-400 mr-3 text-lg">•</span>
                    <span>{gap}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-400 italic">No specific areas for improvement identified.</p>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-slate-600/50"></div>

          {/* Recommendations Section - Now includes both LLM and Skills recommendations */}
          <div>
            <h3 className="text-xl font-bold text-white mb-4">Recommendations</h3>
            <div className="space-y-4">
              {/* LLM Recommendations */}
              {recommendations.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold text-slate-300 mb-3">General Recommendations</h4>
                  <ul className="space-y-3 text-slate-300">
                    {recommendations.map((recommendation: string, index: number) => (
                      <li key={index} className="flex items-start">
                        <span className="text-blue-400 mr-3 text-lg">•</span>
                        <span>{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Skills Recommendations */}
              {skillsRecommendations.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold text-slate-300 mb-3">Skills-Specific Recommendations</h4>
                  <div className="space-y-3">
                    {skillsRecommendations.map((rec: any, index: number) => (
                      <div key={index} className="bg-slate-600/30 rounded-lg p-4 border border-slate-500/30">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="text-md font-semibold text-white">{rec.title}</h5>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            rec.type === 'critical' ? 'bg-red-500/20 text-red-300' :
                            rec.type === 'coverage' ? 'bg-amber-500/20 text-amber-300' :
                            'bg-blue-500/20 text-blue-300'
                          }`}>
                            {rec.type}
                          </span>
                        </div>
                        <p className="text-sm text-slate-300">{rec.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Show message if no recommendations */}
              {recommendations.length === 0 && skillsRecommendations.length === 0 && (
                <p className="text-slate-400 italic">No specific recommendations provided.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 