#!/bin/bash

echo "================================================================================"
echo "                    УСТАНОВКА МОДЕЛЕЙ OLLAMA"
echo "================================================================================"
echo ""

# Проверка что Ollama запущен
echo "Проверка Ollama..."
if ! docker-compose ps ollama | grep -q "Up"; then
    echo "ОШИБКА: Ollama не запущен!"
    echo "Запустите: docker-compose up -d ollama"
    exit 1
fi
echo "Ollama запущен"
echo ""

# Функция установки модели
install_model() {
    local model=$1
    local size=$2

    echo "================================================================================"
    echo "Установка: $model ($size)"
    echo "================================================================================"

    # Проверить установлена ли уже
    if docker-compose exec ollama ollama list | grep -q "$model"; then
        echo "Модель $model уже установлена, пропускаем"
        echo ""
        return
    fi

    echo "Скачивание... (это займёт несколько минут)"
    docker-compose exec ollama ollama pull $model

    if [ $? -eq 0 ]; then
        echo "Модель $model установлена успешно!"
    else
        echo "ОШИБКА: Не удалось установить $model"
    fi
    echo ""
}

# Установка основной модели
echo "================================================================================"
echo "ОСНОВНАЯ МОДЕЛЬ"
echo "================================================================================"
echo ""
echo "Модель: Qwen 2.5 14B Quantized (Q5_K_M)"
echo "Размер: ~9 GB"
echo "RAM требуется: 12 GB"
echo "Профиль: qwen_14b_quantized"
echo ""
echo "Эта модель оптимизирована для систем с 12-16 GB RAM"
echo "Квантизация Q5_K_M обеспечивает 95% качества при половинном размере"
echo ""

install_model "qwen2.5:14b-instruct-q5_k_m" "9 GB"

# ===============================================================================
# АЛЬТЕРНАТИВНЫЕ МОДЕЛИ (раскомментируйте при необходимости)
# ===============================================================================
#
# Для систем с большим количеством RAM или других профилей:
#
# install_model "gpt-oss:20b" "13 GB"        # Для gpt_oss_only профиля (20+ GB RAM)
# install_model "mistral:latest" "4.1 GB"    # Для low_ram профиля (8 GB RAM)
# install_model "llama3:latest" "4.7 GB"     # Для high_quality профиля
# install_model "qwen2.5:7b" "4.7 GB"        # Для balanced профиля
#
# После установки других моделей измените active_profile в config/models.yaml

echo ""
echo "================================================================================"
echo "УСТАНОВЛЕННЫЕ МОДЕЛИ"
echo "================================================================================"
docker-compose exec ollama ollama list
echo ""
echo "================================================================================"
echo "                    УСТАНОВКА ЗАВЕРШЕНА"
echo "================================================================================"
echo ""
echo "Модели сохранены в ./ollama_models/"
echo "При перезапуске контейнера модели останутся на месте"
echo ""
echo "Текущая конфигурация:"
echo "  Профиль: qwen_14b_quantized"
echo "  Модель: qwen2.5:14b-instruct-q5_k_m (для всех агентов)"
echo "  RAM: ~8-9 GB"
echo ""
echo "Проверить конфигурацию:"
echo "  docker-compose exec api python show_config.py"
echo ""
echo "Перезапустить API для применения конфигурации:"
echo "  docker-compose restart api"
echo ""
echo "Запустить обработку статей:"
echo "  docker-compose exec api python run_full_pipeline.py 10"
echo ""
echo "Если нужна другая модель:"
echo "  1. Раскомментируйте нужную модель в этом скрипте"
echo "  2. Запустите ./install_models.sh снова"
echo "  3. Измените active_profile в config/models.yaml"
echo "  4. Перезапустите: docker-compose restart api"
echo ""