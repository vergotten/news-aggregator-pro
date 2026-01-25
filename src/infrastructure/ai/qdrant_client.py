"""
Qdrant Client - векторная БД для дедупликации.
"""

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib


class QdrantService:
    """
    Сервис для работы с Qdrant.
    """
    
    COLLECTION_NAME = "articles"
    VECTOR_SIZE = 384  # Размер вектора для эмбеддингов
    
    def __init__(self, url: str = "http://qdrant:6333"):
        self.client = QdrantClient(url=url)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Создать коллекцию если не существует."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.COLLECTION_NAME for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Коллекция '{self.COLLECTION_NAME}' создана")
        except Exception as e:
            print(f"⚠️  Qdrant warning: {e}")
    
    def text_to_simple_vector(self, text: str) -> List[float]:
        """
        Простой "вектор" из текста (для демо).
        В production используйте Ollama embeddings.
        
        Args:
            text: Текст
            
        Returns:
            Вектор размером VECTOR_SIZE
        """
        # Простой хеш → вектор (для демо)
        hash_val = hashlib.md5(text.encode()).hexdigest()
        
        # Преобразуем в числа
        vector = []
        for i in range(0, len(hash_val), 2):
            val = int(hash_val[i:i+2], 16) / 255.0  # Нормализация
            vector.append(val)
        
        # Дополняем до нужного размера
        while len(vector) < self.VECTOR_SIZE:
            vector.extend(vector[:min(50, self.VECTOR_SIZE - len(vector))])
        
        return vector[:self.VECTOR_SIZE]
    
    def add_article(self, article_id: str, title: str, content: str):
        """
        Добавить статью в Qdrant.
        
        Args:
            article_id: UUID статьи
            title: Заголовок
            content: Контент
        """
        # Создать вектор из заголовка + начало контента
        text = f"{title} {content[:500]}"
        vector = self.text_to_simple_vector(text)
        
        # Сохранить
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=article_id,
                    vector=vector,
                    payload={
                        "title": title,
                        "content_preview": content[:200]
                    }
                )
            ]
        )
    
    def find_similar(
        self,
        title: str,
        content: str,
        limit: int = 5,
        threshold: float = 0.85
    ) -> List[dict]:
        """
        Найти похожие статьи.
        
        Args:
            title: Заголовок
            content: Контент
            limit: Сколько найти
            threshold: Порог схожести (0-1)
            
        Returns:
            Список похожих статей
        """
        text = f"{title} {content[:500]}"
        vector = self.text_to_simple_vector(text)
        
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=vector,
            limit=limit
        )
        
        # Фильтрация по порогу
        similar = []
        for result in results:
            if result.score >= threshold:
                similar.append({
                    'id': result.id,
                    'score': result.score,
                    'title': result.payload.get('title', ''),
                    'content_preview': result.payload.get('content_preview', '')
                })
        
        return similar
    
    def check_duplicate(
        self,
        title: str,
        content: str,
        threshold: float = 0.9
    ) -> Optional[str]:
        """
        Проверить на дубликат.
        
        Args:
            title: Заголовок
            content: Контент
            threshold: Порог дубликата
            
        Returns:
            ID дубликата если найден, иначе None
        """
        similar = self.find_similar(title, content, limit=1, threshold=threshold)
        return similar[0]['id'] if similar else None
