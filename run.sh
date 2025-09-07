#!/bin/bash

ENV_FILE_PATH=.env

[ -f "$ENV_FILE_PATH" ] && set -a && source ".env" && set +a


docker compose down
docker compose up -d

cd triton_inference
python main.py
