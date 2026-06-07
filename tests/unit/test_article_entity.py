"""
Unit tests для Article entity.
"""

import pytest
from src.domain.entities.article import Article
from src.domain.value_objects.source_type import SourceType
from src.domain.value_objects.article_status import ArticleStatus
from src.shared.exceptions.domain_exceptions import DomainValidationError


def test_article_creation():
    """Тест создания статьи."""
    article = Article(
        title="Test Article",
        content="Test content",
        url="https://example.com/test",
        source=SourceType.HABR
    )
    
    assert article.title == "Test Article"
    assert article.content == "Test content"
    assert article.source == SourceType.HABR
    assert article.status == ArticleStatus.PENDING


def test_article_validation_empty_title():
    """Тест валидации - пустой заголовок."""
    with pytest.raises(DomainValidationError):
        Article(
            title="",
            content="Content",
            url="https://example.com",
            source=SourceType.HABR
        )


def test_article_mark_as_news():
    """Тест пометки как новости."""
    article = Article(
        title="News Article",
        content="Content",
        url="https://example.com",
        source=SourceType.HABR
    )
    
    article.mark_as_news("Breaking news")
    
    assert article.is_news is True
    assert article.relevance_reason == "Breaking news"


def test_article_set_relevance():
    """Тест установки релевантности."""
    article = Article(
        title="Article",
        content="Content",
        url="https://example.com",
        source=SourceType.HABR
    )
    
    article.set_relevance(8.5, "High quality content")
    
    assert article.relevance_score == 8.5
    assert article.relevance_reason == "High quality content"


def test_article_invalid_relevance_score():
    """Тест невалидной оценки релевантности."""
    article = Article(
        title="Article",
        content="Content",
        url="https://example.com",
        source=SourceType.HABR
    )
    
    with pytest.raises(DomainValidationError):
        article.set_relevance(15.0, "Invalid score")
