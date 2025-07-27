#!/bin/bash

echo "Setting up AI Resume Reviewer for local development..."

echo
echo "=== Backend Setup ==="
cd backend
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

echo "Starting backend server..."
gnome-terminal --title="Backend Server" -- bash -c "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000; exec bash" &

echo
echo "=== Frontend Setup ==="
cd ../frontend
echo "Installing Node.js dependencies..."
npm install

echo "Starting frontend server..."
gnome-terminal --title="Frontend Server" -- bash -c "npm run dev; exec bash" &

echo
echo "=== Setup Complete ==="
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo
echo "Press Ctrl+C to stop all servers"
wait 