#!/bin/bash
# ISR Demo - Development Setup Script
# Automatically sets up both backend and frontend for local development

set -e

echo "ISR Demo - Development Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Python not found. Please install Python 3.10+${NC}"
    exit 1
fi
python_version=$(python --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python ${python_version}${NC}"

# Check Node
echo -e "${BLUE}Checking Node.js installation...${NC}"
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}Node.js/npm not found. Please install Node.js 18+${NC}"
    exit 1
fi
node_version=$(node --version)
echo -e "${GREEN}✓ Node.js ${node_version}${NC}"

# Backend Setup
echo ""
echo -e "${BLUE}Setting up Backend...${NC}"
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Copy env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env file. Please update it with your configuration.${NC}"
fi

echo -e "${GREEN}✓ Backend setup complete${NC}"

# Frontend Setup
echo ""
echo -e "${BLUE}Setting up Frontend...${NC}"
cd ../frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

echo -e "${GREEN}✓ Frontend setup complete${NC}"

# Final instructions
echo ""
echo -e "${GREEN}======================================"
echo "✓ Setup Complete!${NC}"
echo ""
echo -e "${BLUE}To run the demo:${NC}"
echo ""
echo "1. Terminal 1 - Backend:"
echo "   cd backend"
echo "   source venv/bin/activate  # or venv/Scripts/activate on Windows"
echo "   python main.py"
echo ""
echo "2. Terminal 2 - Frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo -e "${BLUE}Then open:${NC}"
echo "   http://localhost:${VITE_PORT:-5173} (Frontend)"
echo "   http://localhost:${API_PORT:-8123} (Backend)"
echo ""
echo -e "${YELLOW}Note: First run will download models (~15-20 minutes)${NC}"
echo ""
