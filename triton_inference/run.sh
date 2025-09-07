ENV_FILE_PATH=../.env

[ -f "$ENV_FILE_PATH" ] && set -a && source "../.env" && set +a

python main.py

