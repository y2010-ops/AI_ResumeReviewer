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
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-100';
    if (score >= 0.6) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const getRecommendation = (score: number) => {
    if (score >= 0.8) return 'Excellent match! Strong candidate for this position.';
    if (score >= 0.6) return 'Good match. Consider applying with some improvements.';
    return 'Limited match. Focus on developing required skills.';
  };

  const getNextSteps = (score: number) => {
    if (score >= 0.8) return 'Proceed with application and highlight matching skills.';
    if (score >= 0.6) return 'Apply but focus on addressing skill gaps in cover letter.';
    return 'Consider gaining more experience in required areas before applying.';
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
          <button
            onClick={onReset}
            className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            New Analysis
          </button>
        </div>

        {/* Overall Score */}
        <div className="mb-8">
          <div className={`p-6 rounded-lg ${getScoreBgColor(results.final_similarity_score)}`}>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Overall Match Score</h3>
            <div className="flex items-center space-x-4">
              <div className={`text-4xl font-bold ${getScoreColor(results.final_similarity_score)}`}>
                {formatScore(results.final_similarity_score)}
              </div>
              <div className="flex-1">
                <p className="text-gray-700 mb-2">{getRecommendation(results.final_similarity_score)}</p>
                <p className="text-sm text-gray-600">{getNextSteps(results.final_similarity_score)}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Component Scores */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-3">Component Analysis</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Semantic Similarity</span>
                <span className="font-medium">{formatPercentage(results.semantic_score)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Skills Matching</span>
                <span className="font-medium">{formatPercentage(results.skills_score)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">AI Assessment</span>
                <span className="font-medium">{results.llm_score}%</span>
              </div>
              {results.resume_bert_score && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Resume-Specific</span>
                  <span className="font-medium">{formatPercentage(results.resume_bert_score)}</span>
                </div>
              )}
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-3">Skills Analysis</h4>
            {results.skills_analysis && (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Skills Coverage</span>
                  <span className="font-medium">{formatPercentage(results.skills_analysis.coverage_percentage)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Direct Matches</span>
                  <span className="font-medium">{results.skills_analysis.direct_match_count}/{results.skills_analysis.total_job_skills}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Missing Skills</span>
                  <span className="font-medium">{results.skills_analysis.missing_skills?.length || 0}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* AI Analysis Details */}
        {results.llm_details && (
          <div className="space-y-6">
            {results.llm_details.strengths && results.llm_details.strengths.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Key Strengths</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  {results.llm_details.strengths.map((strength: string, index: number) => (
                    <li key={index}>{strength}</li>
                  ))}
                </ul>
              </div>
            )}

            {results.llm_details.gaps && results.llm_details.gaps.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Areas for Improvement</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  {results.llm_details.gaps.map((gap: string, index: number) => (
                    <li key={index}>{gap}</li>
                  ))}
                </ul>
              </div>
            )}

            {results.llm_details.recommendations && results.llm_details.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  {results.llm_details.recommendations.map((rec: string, index: number) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Missing Skills */}
        {results.skills_analysis?.missing_skills && results.skills_analysis.missing_skills.length > 0 && (
          <div className="mt-6">
            <h4 className="font-semibold text-gray-900 mb-3">Missing Skills</h4>
            <div className="flex flex-wrap gap-2">
              {results.skills_analysis.missing_skills.map((skill: string, index: number) => (
                <span key={index} className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm">
                  {skill}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 