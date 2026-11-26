"""Запуск сервера: python run.py"""
import uvicorn

from backend.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("backend.api.main:app", host=s.HOST, port=s.PORT, reload=s.DEBUG)
