"""Создать таблицы: python -m scripts.init_db"""
import asyncio

from backend.db.session import init_db

if __name__ == "__main__":
    asyncio.run(init_db())
    print("Таблицы созданы (файл repf.db)")
