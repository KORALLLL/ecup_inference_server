import os
from typing import Literal

MODEL_NAME: str = os.getenv("TRITON_MODEL_NAME", "bge")
NUMERIC_COLS_TO_ZERO = [
    'rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count',
    'comments_published_count', 'photos_published_count', 'videos_published_count',
    'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30', 'ExemplarAcceptedCountTotal90',
    'OrderAcceptedCountTotal7', 'OrderAcceptedCountTotal30', 'OrderAcceptedCountTotal90',
    'ExemplarReturnedCountTotal7', 'ExemplarReturnedCountTotal30', 'ExemplarReturnedCountTotal90',
    'ExemplarReturnedValueTotal7', 'ExemplarReturnedValueTotal30', 'ExemplarReturnedValueTotal90',
    'ItemVarietyCount', 'ItemAvailableCount',
    'GmvTotal7', 'GmvTotal30', 'GmvTotal90',
]
MODEL_NAME_PRED: str = os.getenv("TRITON_MODEL_NAME_PRED", "backbone")
BEST_THRESHOLD = float(os.getenv("BEST_THRESHOLD", "0.63"))
CHECKPOINTS_PATH = os.getenv("WEIGHTS_FILE", "/home/kirill/ecup_inference_server/weights/models_weights/8000_bert_ftt_imma_BEST.pt")

DEFAULT_BATCH = int(os.getenv("EMBED_BATCH", "64"))
DEFAULT_MAXLEN = int(os.getenv("EMBED_MAXLEN", "512"))

TRITON_PROTOCOL: Literal["http", "grpc"] = os.getenv("TRITON_PROTOCOL", "http")
TRITON_URL: str = os.getenv("TRITON_URL", "http://localhost:8000")
CACHE_PATH = os.getenv("CACHE_PATH", None)
E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-small")