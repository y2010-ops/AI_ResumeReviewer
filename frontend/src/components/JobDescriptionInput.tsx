'use client';

interface JobDescriptionInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function JobDescriptionInput({ value, onChange }: JobDescriptionInputProps) {
  return (
    <div className="w-full space-y-6">
      <div>
      <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
        placeholder="Paste the job description here..."
          className="w-full h-48 px-6 py-4 border-2 border-slate-600/50 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none bg-slate-800/50 backdrop-blur-sm text-white placeholder-slate-400 font-medium leading-relaxed hover:border-slate-500/60 hover:bg-slate-800/60 transition-all duration-200"
        required
      />
      </div>
      
      <div className="bg-gradient-to-br from-slate-800/80 via-slate-700/60 to-slate-800/80 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/50">
        <div className="flex items-start space-x-4">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center flex-shrink-0 mt-1">
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-3">
              <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <p className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 via-teal-300 to-cyan-300 font-bold text-lg">Tips for better analysis:</p>
            </div>
            <p className="text-sm text-slate-300 mb-4">
              Paste the job description or requirements below to analyze how well your resume matches the position.
            </p>
            <ul className="text-slate-200 text-sm space-y-3 leading-relaxed">
              <li className="flex items-start space-x-3 group">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400 font-bold text-lg group-hover:from-emerald-300 group-hover:to-teal-300 transition-all duration-200">•</span>
                <span className="group-hover:text-white transition-colors duration-200">Include specific skills and technologies required</span>
              </li>
              <li className="flex items-start space-x-3 group">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400 font-bold text-lg group-hover:from-teal-300 group-hover:to-cyan-300 transition-all duration-200">•</span>
                <span className="group-hover:text-white transition-colors duration-200">Mention years of experience needed</span>
              </li>
              <li className="flex items-start space-x-3 group">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400 font-bold text-lg group-hover:from-cyan-300 group-hover:to-emerald-300 transition-all duration-200">•</span>
                <span className="group-hover:text-white transition-colors duration-200">Include educational requirements</span>
              </li>
              <li className="flex items-start space-x-3 group">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400 font-bold text-lg group-hover:from-emerald-300 group-hover:to-teal-300 transition-all duration-200">•</span>
                <span className="group-hover:text-white transition-colors duration-200">Add any certifications or specializations</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 