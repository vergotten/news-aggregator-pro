"""
Ollama Client - Infrastructure Layer.
"""

from typing import Optional
import os
import ollama
from src.infrastructure.config.settings import Settings

settings = Settings()


class OllamaClient:
    """
    Клиент для работы с Ollama.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "glm-4.7-flash:q4_K_M")
        self.client = ollama.Client(host=self.base_url)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Генерация текста через Ollama.

        Args:
            prompt: Промпт
            model: Модель (если не указана, используется модель по умолчанию)
            temperature: Температура (0-1)
            max_tokens: Максимум токенов

        Returns:
            Сгенерированный текст
        """
        try:
            # Используем указанную модель или модель по умолчанию
            current_model = model or self.model

            response = self.client.generate(
                model=current_model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    def list_models(self) -> list:
        """Список доступных моделей."""
        try:
            models = self.client.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """
        Загрузить модель, если она не существует.

        Args:
            model: Имя модели

        Returns:
            True если модель успешно загружена или уже существует
        """
        try:
            # Проверяем, существует ли модель
            models = self.list_models()
            if model in models:
                return True

            # Загружаем модель
            self.client.pull(model)
            return True
        except Exception as e:
            print(f"Error pulling model {model}: {e}")
            return False