'use client';

import { useState } from 'react';

export default function JobDescriptionInput() {
  const [jobDescription, setJobDescription] = useState('');

  return (
    <div className="mb-6">
      <label htmlFor="job-description" className="block text-sm font-medium text-gray-700 mb-2">
        Job Description
      </label>
      <textarea
        id="job-description"
        name="job_description"
        rows={8}
        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
        placeholder="Paste the job description here..."
        value={jobDescription}
        onChange={(e) => setJobDescription(e.target.value)}
        required
      />
      <p className="text-xs text-gray-500 mt-1">
        Minimum 50 characters required
      </p>
    </div>
  );
} 