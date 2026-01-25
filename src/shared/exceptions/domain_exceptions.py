"""
Domain Exceptions

Исключения доменного слоя.
"""


class DomainException(Exception):
    """Базовое исключение домена."""
    pass


class DomainValidationError(DomainException):
    """Ошибка валидации доменной сущности."""
    pass


class EntityNotFoundError(DomainException):
    """Сущность не найдена."""
    pass


class DuplicateEntityError(DomainException):
    """Дубликат сущности."""
    pass


class BusinessRuleViolation(DomainException):
    """Нарушение бизнес-правила."""
    pass
