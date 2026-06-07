#!/bin/bash

echo "🤖 Проверка моделей Ollama..."

# Функция проверки модели
check_model() {
    ollama list | grep -q "$1"
}

# Функция установки модели
pull_model() {
    local model=$1
    echo "📥 Установка модели: $model"
    ollama pull $model
    echo "✅ $model установлена"
}

# Ждём запуска Ollama
sleep 5

# Проверка и установка моделей
if ! check_model "mistral"; then
    pull_model "mistral:latest"
else
    echo "✅ mistral уже установлена"
fi

if ! check_model "llama3"; then
    pull_model "llama3:latest"
else
    echo "✅ llama3 уже установлена"
fi

# Опционально: DeepSeek R1 20B
# Раскомментируйте если нужно (требует ~20GB RAM)
# if ! check_model "deepseek-r1"; then
#     pull_model "deepseek-r1:20b"
# else
#     echo "✅ deepseek-r1:20b уже установлена"
# fi

echo "🎉 Все модели готовы!"
