ENV_FILE_PATH=../.env

[ -f "$ENV_FILE_PATH" ] && set -a && source "../.env" && set +a
echo $AWS_URL

aws s3 --endpoint-url "$AWS_URL" sync "$TRITON_REPO_PATH" s3://"$AWS_BUCKET"/triton/model_repo/