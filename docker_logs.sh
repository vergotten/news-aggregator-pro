#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   📋 ЛОГИ КОНТЕЙНЕРОВ                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$1" ]; then
    echo "Доступные сервисы:"
    docker-compose ps --services
    echo ""
    echo "Использование:"
    echo "   ./docker_logs.sh [service]"
    echo "   ./docker_logs.sh all         - все логи"
    echo ""
    echo "Примеры:"
    echo "   ./docker_logs.sh api"
    echo "   ./docker_logs.sh postgres"
    echo "   ./docker_logs.sh ollama"
    exit 0
fi

if [ "$1" == "all" ]; then
    echo "📋 Логи всех сервисов (последние 100 строк):"
    echo ""
    docker-compose logs --tail=100
else
    echo "📋 Логи сервиса: $1 (последние 100 строк)"
    echo "   Нажмите Ctrl+C для выхода"
    echo ""
    docker-compose logs -f --tail=100 "$1"
fi
