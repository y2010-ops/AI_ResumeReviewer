'use client';

interface JobDescriptionInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function JobDescriptionInput({ value, onChange }: JobDescriptionInputProps) {
  return (
    <div className="w-full">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Job Description
      </h2>
      
      <div className="space-y-4">
        <div>
          <label htmlFor="job-description" className="block text-sm font-medium text-gray-700 mb-2">
            Paste the job description here
          </label>
          <textarea
            id="job-description"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Enter the job description, requirements, and responsibilities..."
            className="w-full h-48 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
            required
          />
        </div>
        
        <div className="text-sm text-gray-600">
          <p>💡 Tips for better analysis:</p>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Include specific skills and technologies required</li>
            <li>Mention years of experience needed</li>
            <li>Include educational requirements</li>
            <li>Add any certifications or specializations</li>
          </ul>
        </div>
      </div>
    </div>
  );
} 