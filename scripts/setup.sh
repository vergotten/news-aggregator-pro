#!/bin/bash
# Setup script

echo "🚀 Setting up News Aggregator Pro..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Create .env if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env created. Please edit it with your settings."
fi

# Create project_data directories
echo "📁 Creating project_data directories..."
mkdir -p project_data/{postgres,redis,qdrant,ollama,n8n,directus/uploads,api_logs}

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

echo "✅ Setup complete!"
echo ""
echo "📌 Access points:"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Directus: http://localhost:8055"
echo "   n8n: http://localhost:5678"
echo ""
echo "🔐 Default credentials in docker-compose.yml"
