# Развёртывание и проверки

## Локальная установка

Нужны Python 3.12, доступ к PyPI при установке и права на запись в папку проекта. Команды ниже выполняются в корне `repf` рядом с `run.py`. SQLite создаётся автоматически. Первый запуск генерирует и импортирует базу знаний.

### macOS и Linux

```bash
python3.12 --version
bash scripts/bootstrap.sh
```

Скрипт создаёт окружение `.venv`, устанавливает зависимости, добавляет `.env` при его отсутствии и запускает сервер. Для повторного запуска используйте `.venv/bin/python run.py`.

### Windows · PowerShell

1. Установите Python 3.12. Для обычного ПК с Windows используйте версию x64. Откройте новое окно PowerShell и проверьте, что `py -3.12 --version` показывает Python 3.12. [Документация Python для Windows](https://docs.python.org/3.12/using/windows.html)
2. Откройте PowerShell в папке проекта. Выполните блок по порядку и убедитесь, что каждый шаг завершился успешно.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m scripts.generate_knowledge_base
.\.venv\Scripts\python.exe -m scripts.import_data
.\.venv\Scripts\python.exe run.py
```

Для следующих запусков достаточно `.\.venv\Scripts\python.exe run.py`. Команды используют интерпретатор из `.venv` напрямую. Активировать окружение или менять PowerShell Execution Policy не требуется.

### После запуска

1. Дождитесь `Uvicorn running`. Откройте [127.0.0.1:8000](http://127.0.0.1:8000).
2. Зарегистрируйтесь. Для первого прогона выберите «Временной ряд» → «Взять пример Татарстанстата» → «Рассчитать прогноз».
3. Когда работа закончена, остановите сервер через `Ctrl+C`.

Локальная база `repf.db` содержит аккаунты, настройки и историю. Корпус лежит в `data/processed/`. Для сохранения сессий между рестартами задайте `JWT_SECRET_KEY` длиной от 32 символов в `.env` до первого сохранения ключей.

| Симптом | Что проверить |
|---|---|
| Windows сообщает, что `py` не найден | Установлен ли Python с лаунчером. После установки откройте новое окно PowerShell |
| Не находится `.venv\Scripts\python.exe` | Открыт ли корень проекта и завершилась ли команда `py -3.12 -m venv .venv` |
| Порт 8000 занят | Остановите предыдущий экземпляр REPF или смените `PORT` в `.env`, затем откройте указанный порт |
| Прогноз завершился ошибкой | Проверьте статус задачи в разделе «История» и `logs/app.log` |

## Qdrant при локальном запуске

Чтобы подключить Qdrant к Python-версии REPF, из корня проекта запустите контейнер. Команда одинакова для PowerShell, macOS и Linux.

```bash
docker run -d --name repf-local-qdrant --restart unless-stopped -p 127.0.0.1:6333:6333 -v repf-local-qdrant:/qdrant/storage qdrant/qdrant:v1.13.4
```

Добавьте в `.env` строку `QDRANT_URL=http://127.0.0.1:6333`. Опубликуйте векторы командой `.venv/bin/python -m scripts.index_qdrant` или `.\.venv\Scripts\python.exe -m scripts.index_qdrant` в PowerShell и перезапустите REPF. После обновления корпуса повторите публикацию векторов.

Запустите прогноз в режиме «Подбор по аналогам». В «Трассировке агентов» откройте «Поиск кейсов» и проверьте поле `backend`. При успешном поиске через Qdrant там указано `qdrant`. Если сервис недоступен или индекс отстаёт от корпуса, REPF использует `local_pca`. Для запуска уже созданного контейнера после ручной остановки используйте `docker start repf-local-qdrant`. Полный Docker Compose настраивает Qdrant и публикует индекс самостоятельно.

## LLM

В интерфейсе откройте «Настройки» после входа. В аккаунте можно сохранить несколько именованных пресетов с отдельными моделями и API-ключами. В списке серверов OpenAI-совместимого API доступны DeepSeek, OpenAI, OpenRouter и Groq. Чтобы добавить собственный адрес, задайте на сервере `COMPATIBLE_LLM_URL`. Модель получает агрегированные признаки ряда и найденные кейсы для выбора инструмента. Сохранение, переключение и кнопка «Проверить» выполняют по одному пробному вызову инструмента для выбранной модели. При ошибке сохранение или переключение отклоняется. [OpenAI](https://platform.openai.com/docs/api-reference/chat), [OpenRouter](https://learn.openrouter.ai/courses/first-request/make-a-request), [Groq](https://console.groq.com/docs/api-reference)

Ключи аккаунтов шифруются в базе. Для локального запуска ключ шифрования создаётся в `.repf-llm.key` в каталоге проекта, если не задан достаточно длинный `JWT_SECRET_KEY`. В Docker используется ключ, производный от `JWT_SECRET_KEY`. При смене ключа шифрования сохранённые API-ключи нужно ввести заново. Для локальной резервной копии сохраняйте базу и файл ключа вместе в защищённом месте.

| Переменная | Значение |
|---|---|
| `LLM_PROVIDER` | `auto`, `deepseek`, `ollama`, `compatible` или `mock` |
| `LLM_API_KEY` | Ключ выбранного облачного провайдера. В `auto` используется для совместимого API при заданном `COMPATIBLE_LLM_URL`, иначе для DeepSeek |
| `DEEPSEEK_API_KEY`, `COMPATIBLE_LLM_KEY` | Старые настройки для конкретных провайдеров. Они по-прежнему поддерживаются |
| `DEEPSEEK_MODEL` | `deepseek-flash` по умолчанию |
| `OLLAMA_URL`, `LLM_MODEL` | Базовый адрес Ollama и имя установленной модели |
| `COMPATIBLE_LLM_URL`, `COMPATIBLE_LLM_MODEL` | Адрес совместимого API с `/v1` и модель vLLM или другого провайдера |
| `LLM_TIMEOUT` | Таймаут одного вызова в секундах |

Храните ключ в локальном `.env`, исключённом из Git и Docker-контекста. При переключении провайдера обновите значение `LLM_API_KEY`. Для Ollama общий ключ не используется.

`.venv/bin/python -m scripts.verify_deepseek` проверяет передачу схемы, нативный tool call и локальную валидацию. Он выводит имя провайдера, статус и расход токенов. Нативный вызов подтверждается только при успешной проверке возвращённого инструмента. В PowerShell выполните `.\.venv\Scripts\python.exe -m scripts.verify_deepseek`.

## Docker Compose

На macOS и Windows запустите Docker Desktop. Для Windows используйте Linux-контейнеры и подключение через WSL 2. На Linux нужны Docker Engine и плагин Compose. Проверьте установку командой `docker compose version`. Для обычного локального запуска через Python Docker не нужен. [Docker Desktop для Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

1. Создайте `.env` из шаблона, задайте уникальный `JWT_SECRET_KEY` от 32 символов и пароль `POSTGRES_PASSWORD` в безопасном для URL формате. Два независимых значения можно получить, дважды запустив `python -c "import secrets; print(secrets.token_hex(32))"` под вашим Python. На Windows используйте `py -3.12 -c "import secrets; print(secrets.token_hex(32))"`.
2. Выполните `docker compose up --build -d`.
3. Дождитесь завершения сервиса `init` и здоровья `api`. Откройте [localhost:8080](http://localhost:8080).
4. Для своего Ollama в Compose задайте `OLLAMA_URL=http://ollama:11434` в `.env` и выполните `docker compose --profile ollama up -d`. Затем загрузите модель командой `docker compose exec ollama ollama pull qwen2.5:7b`. Для Ollama на хосте задайте `OLLAMA_URL=http://host.docker.internal:11434`.

`docker compose ps -a` показывает состояние сервисов. `docker compose logs -f init api worker scheduler` показывает инициализацию и очередь. База данных, индекс Qdrant, корпус и модели Ollama сохраняются в томах Compose.

Администратора можно создать на первом запуске через `ADMIN_EMAIL` и `ADMIN_PASSWORD`. Для существующего аккаунта применяется `.venv/bin/python -m scripts.manage_user email@example.org editor`. В PowerShell используйте `.\.venv\Scripts\python.exe -m scripts.manage_user email@example.org editor`. В Docker ту же команду запускают через `docker compose exec api python -m scripts.manage_user email@example.org editor`.

## Проверки

На macOS и Linux.

```bash
.venv/bin/python -m pytest --cov=backend --cov-report=term-missing --cov-fail-under=70
node --check frontend/app.js
docker compose config --quiet
```

В PowerShell.

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=backend --cov-report=term-missing --cov-fail-under=70
```

Тестовая конфигурация использует временную БД и моки LLM. Логи приложения в `logs/app.log`, сценарий оценки пишет в `logs/eval.log`. Схема CI находится в `.github/workflows/ci.yml`.
