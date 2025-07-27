@echo off
echo 🚀 Starting AI Resume-Job Matcher Development Environment

echo 📦 Installing frontend dependencies...
cd frontend
call npm install

echo 🐍 Installing backend dependencies...
cd ..\backend
call pip install -r requirements.txt

echo 🔧 Starting backend server...
cd ..
start "Backend Server" cmd /k "cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo 🎨 Starting frontend development server...
cd frontend
start "Frontend Server" cmd /k "npm run dev"

echo ✅ Development environment started!
echo 🌐 Frontend: http://localhost:3000
echo 🔗 Backend API: http://localhost:8000
echo.
echo Press any key to stop servers...
pause >nul

echo 🛑 Stopping servers...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ Servers stopped 