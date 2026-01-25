"""
Ollama Client - Infrastructure Layer.
"""

from typing import Optional
import ollama
from src.infrastructure.config.settings import Settings

settings = Settings()


class OllamaClient:
    """
    Клиент для работы с Ollama.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url
        self.client = ollama.Client(host=self.base_url)
    
    def generate(
        self,
        prompt: str,
        model: str = "mistral:latest",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Генерация текста через Ollama.
        
        Args:
            prompt: Промпт
            model: Модель (mistral, llama3, deepseek-r1:20b)
            temperature: Температура (0-1)
            max_tokens: Максимум токенов
            
        Returns:
            Сгенерированный текст
        """
        try:
            response = self.client.generate(
                model=model,
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
