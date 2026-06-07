#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           🔄 БЫСТРЫЙ ПЕРЕЗАПУСК (БЕЗ REBUILD)                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

echo "🛑 Остановка..."
docker-compose down
echo ""

echo "🚀 Запуск..."
docker-compose up -d
echo ""

echo "⏳ Ожидание (10 сек)..."
sleep 10
echo ""

echo "📊 Статус:"
docker-compose ps
echo ""

echo "✅ Готово!"
echo "   http://localhost:8000"
