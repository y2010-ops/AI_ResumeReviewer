'use client';

interface AnalysisResultsProps {
  results: any;
  onReset: () => void;
}

export default function AnalysisResults({ results, onReset }: AnalysisResultsProps) {
  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatScore = (value: number) => `${(value * 100).toFixed(1)}%`;

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    if (score >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-100';
    if (score >= 0.6) return 'bg-yellow-100';
    if (score >= 0.4) return 'bg-orange-100';
    return 'bg-red-100';
  };

  const getRecommendation = (score: number) => {
    if (score >= 0.8) return 'Strongly Recommended';
    if (score >= 0.6) return 'Recommended with Minor Concerns';
    if (score >= 0.4) return 'Consider with Development Plan';
    return 'Not Recommended';
  };

  const getNextSteps = (score: number) => {
    if (score >= 0.8) return 'Proceed to interview';
    if (score >= 0.6) return 'Schedule technical assessment';
    if (score >= 0.4) return 'Request additional training/certifications';
    return 'Consider alternative roles or candidates';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
          <button
            onClick={onReset}
            className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
          >
            New Analysis
          </button>
        </div>

        {/* Final Match Score */}
        <div className="text-center mb-6">
          <div className={`inline-flex items-center px-6 py-3 rounded-full ${getScoreBgColor(results.final_similarity_score)}`}>
            <span className={`text-3xl font-bold ${getScoreColor(results.final_similarity_score)}`}>
              {formatScore(results.final_similarity_score)}
            </span>
          </div>
          <p className="text-lg font-medium text-gray-700 mt-2">
            {results.similarity_category}
          </p>
        </div>
      </div>

      {/* Feedback/Analysis - LLM Insights */}
      {results.llm_details && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">📋 AI Analysis & Feedback</h3>
          
          {/* Strengths */}
          {results.llm_details.strengths && results.llm_details.strengths.length > 0 && (
            <div className="mb-4">
              <h4 className="text-lg font-medium text-gray-900 mb-2">✅ Key Strengths</h4>
              <ul className="space-y-1">
                {results.llm_details.strengths.slice(0, 3).map((strength: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    <span className="text-gray-700">{strength}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Gaps */}
          {results.llm_details.gaps && results.llm_details.gaps.length > 0 && (
            <div className="mb-4">
              <h4 className="text-lg font-medium text-gray-900 mb-2">⚠️ Areas for Improvement</h4>
              <ul className="space-y-1">
                {results.llm_details.gaps.slice(0, 3).map((gap: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-orange-500 mr-2">•</span>
                    <span className="text-gray-700">{gap}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommendations */}
          {results.llm_details.recommendations && results.llm_details.recommendations.length > 0 && (
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">💡 Recommendations</h4>
              <ul className="space-y-1">
                {results.llm_details.recommendations.slice(0, 3).map((rec: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Skills Match and Missing Skills */}
      {results.skills_analysis && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">🔧 Skills Analysis</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-green-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-green-800 mb-1">Skills Match</h4>
              <p className="text-2xl font-bold text-green-600">
                {results.skills_analysis.direct_match_count}/{results.skills_analysis.total_job_skills}
              </p>
              <p className="text-sm text-green-700">
                ({formatPercentage(results.skills_analysis.direct_coverage)})
              </p>
            </div>
            <div className="bg-yellow-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-yellow-800 mb-1">Missing Skills</h4>
              <p className="text-2xl font-bold text-yellow-600">
                {results.skills_analysis.missing_skills?.length || 0}
              </p>
              <p className="text-sm text-yellow-700">Critical gaps</p>
            </div>
          </div>

          {/* Matched Skills */}
          {results.skills_analysis.direct_matches && results.skills_analysis.direct_matches.length > 0 && (
            <div className="mb-4">
              <h4 className="text-lg font-medium text-gray-900 mb-2">✅ Matched Skills</h4>
              <div className="flex flex-wrap gap-2">
                {results.skills_analysis.direct_matches.map((skill: string, index: number) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Missing Skills */}
          {results.skills_analysis.missing_skills && results.skills_analysis.missing_skills.length > 0 && (
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">❌ Missing Skills</h4>
              <div className="flex flex-wrap gap-2">
                {results.skills_analysis.missing_skills.map((skill: string, index: number) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Summary */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg shadow-lg p-6 text-white">
        <h3 className="text-xl font-semibold mb-4">📋 Summary</h3>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>Final Score</span>
            <span className="font-bold">{formatScore(results.final_similarity_score)} ({results.similarity_category})</span>
          </div>
          <div className="flex justify-between">
            <span>Recommendation</span>
            <span className="font-bold">{getRecommendation(results.final_similarity_score)}</span>
          </div>
          <div className="flex justify-between">
            <span>Next Steps</span>
            <span className="font-bold">{getNextSteps(results.final_similarity_score)}</span>
          </div>
        </div>
      </div>
    </div>
  );
} 