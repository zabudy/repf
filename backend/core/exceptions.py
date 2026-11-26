"""Базовые исключения проекта."""


class REPFError(Exception):
    """Базовая ошибка приложения."""


class DataValidationError(REPFError):
    """Данные не прошли проверку качества."""


class ForecastError(REPFError):
    """Не удалось построить прогноз."""
