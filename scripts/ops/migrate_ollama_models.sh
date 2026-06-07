#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║      🔄 МИГРАЦИЯ МОДЕЛЕЙ OLLAMA В ЛОКАЛЬНУЮ ДИРЕКТОРИЮ          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Проверка что мы в правильной директории
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Ошибка: docker-compose.yml не найден"
    echo "   Запустите скрипт из директории news-aggregator-pro"
    exit 1
fi

echo "1️⃣  Проверка старого volume..."
if docker volume ls | grep -q "news-aggregator-pro_ollama_data"; then
    echo "   ✅ Найден volume: news-aggregator-pro_ollama_data"
    VOLUME_EXISTS=true
else
    echo "   ℹ️  Старый volume не найден (возможно уже мигрировали)"
    VOLUME_EXISTS=false
fi

echo ""
echo "2️⃣  Создание директории ollama_models..."
mkdir -p ollama_models
echo "   ✅ Директория создана/существует"

echo ""
if [ "$VOLUME_EXISTS" = true ]; then
    echo "3️⃣  Копирование моделей из volume..."
    echo "   ⏳ Это может занять 1-2 минуты..."
    
    docker run --rm \
      -v news-aggregator-pro_ollama_data:/source \
      -v $(pwd)/ollama_models:/destination \
      alpine sh -c "cp -r /source/* /destination/" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Модели скопированы"
    else
        echo "   ⚠️  Предупреждение: Копирование завершилось с ошибками"
        echo "   Попробуйте скачать модели заново после перезапуска"
    fi
else
    echo "3️⃣  Копирование пропущено (volume не найден)"
fi

echo ""
echo "4️⃣  Проверка содержимого..."
if [ -d "ollama_models/models" ]; then
    SIZE=$(du -sh ollama_models/ 2>/dev/null | cut -f1)
    echo "   ✅ Размер: $SIZE"
else
    echo "   ℹ️  Директория пуста (скачаем модели после запуска)"
fi

echo ""
echo "5️⃣  Перезапуск с новыми настройками..."
docker-compose down 2>/dev/null
docker-compose up -d ollama

echo "   ⏳ Ждём запуска Ollama (10 секунд)..."
sleep 10

echo ""
echo "6️⃣  Проверка моделей..."
if docker-compose exec ollama ollama list 2>/dev/null | grep -q "NAME"; then
    echo ""
    echo "   📊 МОДЕЛИ В OLLAMA:"
    echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker-compose exec ollama ollama list
    echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    MODEL_COUNT=$(docker-compose exec ollama ollama list | tail -n +2 | wc -l)
    
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "   ✅ Найдено моделей: $MODEL_COUNT"
        MIGRATION_SUCCESS=true
    else
        echo "   ⚠️  Модели не найдены"
        MIGRATION_SUCCESS=false
    fi
else
    echo "   ⚠️  Не удалось подключиться к Ollama"
    MIGRATION_SUCCESS=false
fi

echo ""
if [ "$MIGRATION_SUCCESS" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ МИГРАЦИЯ УСПЕШНА!                         ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📂 Модели теперь в: ./ollama_models/"
    echo ""
    
    if [ "$VOLUME_EXISTS" = true ]; then
        echo "🗑️  Удалить старый volume? (рекомендуется)"
        read -p "   Удалить news-aggregator-pro_ollama_data? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm news-aggregator-pro_ollama_data 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "   ✅ Старый volume удалён"
            else
                echo "   ⚠️  Не удалось удалить (возможно уже удалён)"
            fi
        else
            echo "   ℹ️  Volume сохранён (можете удалить позже вручную)"
        fi
    fi
else
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║              ⚠️  НУЖНО СКАЧАТЬ МОДЕЛИ                          ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Модели не найдены. Скачайте их:"
    echo ""
    echo "   docker-compose exec ollama ollama pull mistral:latest"
    echo "   docker-compose exec ollama ollama pull llama3:latest"
    echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 ТЕПЕРЬ МОДЕЛИ:"
echo "   ✅ Сохраняются в ./ollama_models/"
echo "   ✅ НЕ удаляются при docker-compose down -v"
echo "   ✅ Легко делать бэкап"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
