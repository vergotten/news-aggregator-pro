"""
Infrastructure Exceptions

Исключения инфраструктурного слоя.
"""


class InfrastructureException(Exception):
    """Базовое исключение инфраструктуры."""
    pass


class DatabaseError(InfrastructureException):
    """Ошибка работы с БД."""
    pass


class ExternalServiceError(InfrastructureException):
    """Ошибка внешнего сервиса."""
    pass


class CacheError(InfrastructureException):
    """Ошибка работы с кэшем."""
    pass


class MessageQueueError(InfrastructureException):
    """Ошибка работы с очередью сообщений."""
    pass
