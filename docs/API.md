# REST API

Интерактивная схема доступна в `/docs`. Базовый адрес локального сервера — `http://127.0.0.1:8000`.

## Авторизация

`POST /auth/register` и `POST /auth/login` принимают JSON с полями `email` и `password`. Ответ содержит `access_token`. Передавайте его в заголовке `Authorization: Bearer <token>`. Профиль доступен в `GET /auth/me`.

Пользователь просматривает свои задачи и может принять или пересчитать собственный результат. Роли редактора и администратора видят очередь проверки. Администратор может читать любую задачу и запускать оценку с подключённой внешней LLM.

## Прогноз

| Метод и путь | Ответ |
|---|---|
| `POST /api/v1/forecast/` | `202 Accepted`, `task_id`, начальный статус |
| `GET /api/v1/forecast/` | Последние 50 задач. Параметр `review_queue=true` открывает очередь HITL |
| `GET /api/v1/forecast/{task_id}/status` | Статус, доступные инструменты, горизонт, решение и возможная ошибка |
| `GET /api/v1/forecast/{task_id}/result` | Прогноз, интервалы, модель, метрики, кейсы, вывод Critic, `chat_enabled` и ссылка на скачивание |
| `POST /api/v1/forecast/{task_id}/chat` | Ответ LLM о результате прогноза и имя модели |
| `GET /api/v1/forecast/{task_id}/download` | CSV с датой, прогнозом и границами интервала |
| `GET /api/v1/forecast/{task_id}/reasoning` | Дерево событий с дочерними шагами и вызовами LLM |
| `POST /api/v1/forecast/{task_id}/hitl` | Ручное решение по задаче в `waiting_for_human` |
| `GET /api/v1/forecast/demo/series` | Открытый пример ряда Татарстанстата |

Для загрузки CSV можно использовать curl.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/forecast/ \
  -H "Authorization: Bearer $REPF_TOKEN" \
  -F file=@data/import/tatarstan_secondary.csv \
  -F frequency=QS -F horizon=3 -F mode=+agents
```

Для Windows PowerShell присвойте токен переменной окружения `REPF_TOKEN` и используйте `curl.exe`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/forecast/ -H "Authorization: Bearer $env:REPF_TOKEN" -F "file=@data/import/tatarstan_secondary.csv" -F "frequency=QS" -F "horizon=3" -F "mode=+agents"
```

В запросе оценки квартиры передайте параметры объекта.

```json
{
  "data": {
    "task_type": "tabular",
    "apartment": {
      "district": "Вахитовский",
      "area_total": 54,
      "floor": 5,
      "floors_total": 9,
      "build_year": 1985,
      "house_type": "panel",
      "condition": "cosmetic"
    }
  },
  "horizon": 6,
  "mode": "+agents"
}
```

JSON-загрузка ряда использует `data.task_type=timeseries` и `data.observations=[{"date":"2024-01-01","value":100}]` с необходимым количеством наблюдений. Допустимы поля `frequency`, `seasonal_period` и `nonnegative`. Предел тела запроса — 5 МБ.

HITL принимает `{"action":"accept","comment":"Проверено"}`, `{"action":"retry","model":"fit_holt","horizon":4}` или `{"action":"reject","comment":"Нужны дополнительные данные"}`. Для повтора достаточно изменить один параметр. Результат возвращается в статусах `completed` и `waiting_for_human`. Недоступный результат возвращает 409.

Чат доступен владельцу готового результата в режиме `+agents`. Передайте `{"question":"Как читать прогнозный интервал?","history":[]}`. В необязательном поле `history` можно передать до шести предыдущих пар `{"question":"…","answer":"…"}`. Максимальная длина вопроса — 1500 символов. При успехе API возвращает поля `answer` и `model`. Код 503 означает, что требуется настроить доступную модель, а 502 — что выбранная модель не ответила.

## Эксперименты

- `POST /api/v1/evaluate/batch` создаёт задачу с параметрами `dataset`, `num_series`, `horizons`, `data_fractions`, `modes`. Ответ содержит `evaluation_id`.
- `GET /api/v1/evaluate/compare?evaluation_id=<id>` возвращает состояние и после завершения таблицу качества и парные сравнения. Без идентификатора открывает последний эксперимент пользователя.
- `GET /api/v1/glossary?query=MASE` ищет определения в импортированном справочнике.

## Настройки модели

| Метод и путь | Действие |
|---|---|
| `GET /api/v1/llm/settings` | Возвращает настройки аккаунта, список пресетов `checkpoints` и выбранный `active_checkpoint_id` |
| `POST /api/v1/llm/checkpoints` | Проверяет доступность модели и сохраняет пресет с обязательным `name` |
| `PUT /api/v1/llm/selection` | Проверяет и активирует `checkpoint_id`. Значение `null` выбирает настройки сервера |
| `POST /api/v1/llm/checkpoints/{id}/probe` | Повторно проверяет реальный вызов инструмента для пресета |
| `DELETE /api/v1/llm/checkpoints/{id}` | Удаляет свой пресет. Удаление активного возвращает настройки сервера |
| `PUT /api/v1/llm/settings` | Проверяет и сохраняет одиночный профиль старого API |
| `POST /api/v1/llm/probe` | Проверяет один нативный вызов инструмента у сохранённой модели |
| `DELETE /api/v1/llm/settings` | Удаляет профиль старого API и выбирает настройки сервера. Именованные пресеты сохраняются |

`provider` принимает `deepseek`, `compatible`, `ollama` или `local`. Для `compatible` требуются имя модели и один из адресов, возвращаемых GET. В устаревшем `PUT /settings` отсутствие `api_key` оставляет прежний ключ для того же провайдера и сервера. Пустая строка удаляет ключ у совместимого API. Для нового пресета укажите собственный ключ, если провайдер его требует. Все операции требуют авторизации. API никогда не возвращает сохранённый ключ.

У нового облачного или Ollama-пресета обязательны `name` и точный идентификатор `model`. Сохранение пресета оставляет выбор активной модели прежним. При недоступности провайдера, неизвестной модели или некорректном Tool Calling ручной выбор и создание отклоняются. Настройка из старого API отображается как «Ранее сохранённая модель».
