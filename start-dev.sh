#!/bin/bash

echo "🚀 Starting AI Resume-Job Matcher Development Environment"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check if backend port is available
if ! check_port 8000; then
    echo "❌ Backend port 8000 is already in use. Please stop the existing service."
    exit 1
fi

# Check if frontend port is available
if ! check_port 3000; then
    echo "❌ Frontend port 3000 is already in use. Please stop the existing service."
    exit 1
fi

echo "📦 Installing frontend dependencies..."
cd frontend
npm install

echo "🐍 Installing backend dependencies..."
cd ../backend
pip install -r requirements.txt

echo "🔧 Starting backend server..."
cd ..
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 5

echo "🎨 Starting frontend development server..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ Development environment started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
wait

# Cleanup
echo "🛑 Stopping servers..."
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
echo "✅ Servers stopped" 