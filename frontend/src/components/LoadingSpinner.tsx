export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center space-x-3">
      <div className="relative">
        <div className="w-6 h-6 border-2 border-blue-200 rounded-full"></div>
        <div className="absolute top-0 left-0 w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
      <span className="text-blue-600 font-semibold">Analyzing...</span>
    </div>
  );
} 