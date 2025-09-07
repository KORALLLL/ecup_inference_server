ENV_FILE_PATH=.env

[ -f "$ENV_FILE_PATH" ] && set -a && source ".env" && set +a

# conda create -n ecup python=3.10 --no-default-packages -y
# conda activate ecup
# pip install uv

cd onnx_convertation
# uv pip install -r requirements.txt

python get_config_combined.py
python export_combined_onnx.py

cd ..
docker compose down
docker compose up -d