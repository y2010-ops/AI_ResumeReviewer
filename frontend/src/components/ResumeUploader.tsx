'use client';

import { useState, useRef } from 'react';

interface ResumeUploaderProps {
  onFileSelect: (file: File) => void;
}

export default function ResumeUploader({ onFileSelect }: ResumeUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/pdf') {
        setSelectedFile(file);
        onFileSelect(file);
      }
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type === 'application/pdf') {
        setSelectedFile(file);
        onFileSelect(file);
      }
    }
  };

  const handleButtonClick = () => {
    inputRef.current?.click();
  };

  const removeFile = () => {
    setSelectedFile(null);
    onFileSelect(null as any);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Upload Resume (PDF)
      </h2>
      
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-blue-400 bg-blue-50' 
            : selectedFile 
              ? 'border-green-400 bg-green-50' 
              : 'border-gray-300 bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileInputChange}
          className="hidden"
        />
        
        {!selectedFile ? (
          <div>
            <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="text-gray-600 mb-2">
              Drag and drop your resume here, or
            </p>
            <button
              type="button"
              onClick={handleButtonClick}
              className="text-blue-600 hover:text-blue-700 font-medium"
            >
              browse to select file
            </button>
            <p className="text-sm text-gray-500 mt-2">
              Only PDF files are supported
            </p>
          </div>
        ) : (
          <div>
            <svg className="mx-auto h-12 w-12 text-green-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-green-600 font-medium mb-2">
              {selectedFile.name}
            </p>
            <p className="text-sm text-gray-500 mb-4">
              File size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
            <button
              type="button"
              onClick={removeFile}
              className="text-red-600 hover:text-red-700 text-sm font-medium"
            >
              Remove file
            </button>
          </div>
        )}
      </div>
    </div>
  );
} 