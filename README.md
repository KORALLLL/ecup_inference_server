## README.md

# ecup_inference_server

Решение команды **lab260** для хакатона ecup в треке «Контроль качества». Сервис поднимает Triton Inference Server с модельным репозиторием и API на FastAPI для end-to-end инференса: загрузка CSV, извлечение эмбеддингов и предсказания с выгрузкой результатов.

## Состав

- Triton Inference Server (HTTP/gRPC + метрики) с подключённым репозиторием моделей.
- FastAPI-сервис для приёма CSV, вызова Triton и сохранения/выдачи предсказаний.
- Docker Compose для локального развёртывания и межсервисной сети.

## Подготовка весов

Перед запуском распаковать архив с весами в каталог weights в корне репозитория:

- Скачать архив models_weights.tar по ссылке: https://disk.yandex.com/d/7RCEEjCOy1KMGQ.
- Распаковать в папку weights так, чтобы получилась структура:

```
weights/
└── models_weights
    ├── 8000_bert_ftt_imma_BEST.pt
    └── models--BAAI--bge-m3
```

Убедиться, что итоговые пути внутри контейнеров будут доступны по /weights/models_weights/... при соответствующем монтировании volumes.

Пример распаковки в Linux/macOS:
```bash
mkdir -p weights
tar -xf models_weights.tar -C weights
```

## Конфигурация окружения

В корне репозитория есть файл .env.example — скопировать его в .env и при необходимости отредактировать значения:

```bash
cp .env.example .env
```

Ключевые переменные:
- TRITON_REPO_PATH — путь к локальному репозиторию моделей для Triton (монтируется в контейнер как /models).
- WEIGHTS_PATH — локальный путь к каталогу weights (монтируется как /weights).
- TRITON_MODEL_NAME, TRITON_MODEL_NAME_PRED — имена моделей в репозитории Triton.
- TRITON_PROTOCOL — http или grpc для клиента API.
- TRITON_URL — адрес Triton внутри Compose-сети: triton:8000 для HTTP или triton:8001 для gRPC (без схемы).
- Прочие параметры инференса и батчинга см. в .env.example.

Примечание: Docker Compose по умолчанию подставляет переменные из .env, можно проверить итоговую конфигурацию командой docker compose config.

## Запуск

1) Собрать и запустить сервисы:
```bash
docker compose up -d --build
```

2) Проверить готовность Triton:
- HTTP: внутри сети Compose доступен по http://triton:8000/v2/health/ready.
- gRPC: порт 8001 внутри сети, наружные порты проброшены из compose.yml.

3) Открыть API: FastAPI по умолчанию на http://localhost:8001, Swagger UI — http://localhost:8001/docs.

## Использование API

- Эндпоинт для полного конвейера: POST /run_full принимает CSV как form-data (поле file) и возвращает файл с предсказаниями в виде скачивания.
- Пример запроса curl:
```bash
curl -X POST http://localhost:8001/run_full \
  -F "file=@/path/to/data.csv" \
  -F "batch=128" -F "clean=true" -F "e5_maxlen=512" \
  -o preds.csv
```

## Сетевые нюансы Compose

- Между контейнерами использовать имя сервиса как хост: triton, и внутренние порты 8000 (HTTP) / 8001 (gRPC). Наружные 18000/18001 нужны только для доступа с хоста.
- Для клиентов Triton указывать адрес без схемы: host:port. Для HTTP — triton:8000; для gRPC — triton:8001.

## Траблшутинг

- Если API не может подключиться к Triton по gRPC с ошибкой UNAVAILABLE/Socket closed — проверьте, что указан порт 8001 и Triton успел загрузиться.
- Если torch.load не находит чекпоинт — убедитесь, что переменная WEIGHTS_FILE указывает на корректный путь внутри контейнера, например /weights/models_weights/8000_bert_ftt_imma_BEST.pt, и том weights смонтирован.
- Валидируйте .env подстановки: docker compose config — покажет итоговые значения.
