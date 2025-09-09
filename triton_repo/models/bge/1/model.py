import os,torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"  # не ходить в сеть [6]
torch.set_num_threads(1)

import re
import argparse
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

tqdm.pandas()

import torch.nn as nn
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel

import numpy as np
import triton_python_backend_utils as pb_utils

torch.backends.cuda.enable_flash_sdp(True)


class TritonPythonModel:
    def initialize(self, args):
        # Путь к модели BGE из env/по умолчанию
        kind = args.get("model_instance_kind")
        dev_id = args.get("model_instance_device_id")
        self.bge_path = os.getenv(
            "BGE_PATH",
            "/weights/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
        )
        # Параметры кодирования (можно переопределить переменными окружения)
        self.use_fp16 = os.getenv("USE_FP16", "1") == "1"
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.batch_size = int(os.getenv("ENC_BATCH", "64"))

        if kind == "KIND_GPU" and (dev_id is not None) and torch.cuda.is_available():
            dev_idx = int(dev_id)
            torch.cuda.set_device(dev_idx)  # критично для multi-GPU инстансов
            self.device = torch.device(f"cuda:{dev_idx}")
        else:
            self.device = torch.device("cpu")
            self.use_fp16 = False

        self.model = BGEM3FlagModel(self.bge_path, use_fp16=self.use_fp16, device=self.device)


        # Тип выхода: FP16 или FP32
        self._out_dtype = np.float16

    def execute(self, requests):
        # 1) Собираем тексты из всех запросов
        all_texts = []
        slices = []
        total = 0

        for req in requests:
            # Вход должен называться "TEXT" и быть BYTES/TYPE_STRING
            in_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
            if in_tensor is None:
                return [pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Missing required input TEXT"))
                ]
            # Превращаем в список строк (UTF-8)
            np_bytes = in_tensor.as_numpy()  # dtype=object или bytes [6]
            # Поддержка как bytes, так и object-строк
            if np_bytes.dtype.kind in ("S", "U", "O"):
                # Если bytes -> декодируем; если уже str -> оставляем
                batch_texts = []
                for item in np_bytes.reshape(-1):
                    if isinstance(item, bytes):
                        batch_texts.append(item.decode("utf-8", errors="ignore"))
                    else:
                        batch_texts.append(str(item))
            else:
                return [pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"TEXT must be BYTES/STRING, got dtype={np_bytes.dtype}"))
                ]

            B = len(batch_texts)
            all_texts.extend(batch_texts)
            slices.append((total, total + B))
            total += B

        # 2) Один вызов encode для всего пула (батчами внутри)
        if len(all_texts) == 0:
            # Пустой ответ
            return [pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("dense_vecs", np.zeros((0, 0), dtype=self._out_dtype))
            ])]

        with torch.no_grad():
            out = self.model.encode(
                all_texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            dense = out["dense_vecs"]  # List[List[float]] или np.ndarray [7][10]
            dense = np.asarray(dense, dtype=self._out_dtype)

        # 3) Режем обратно по запросам и формируем ответы
        responses = []
        for s, e in slices:
            part = dense[s:e]  # [B_i, H]
            out_tensor = pb_utils.Tensor("dense_vecs", part)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        pass


MAX_LEN_TEXT = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "lxml").get_text(" ")
    text = text.lower()
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_embeddings(bge_model: BGEM3FlagModel, texts, batch_size=64, max_length=512):
    embeddings = []
    with torch.no_grad():
        outputs = bge_model.encode(texts, batch_size=batch_size, max_length=max_length)['dense_vecs']
    embeddings.extend(outputs)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV, compute BGEM3 embeddings, and run inference with E5+FTT model")
    parser.add_argument('--first_csv', default="/home/kirill/ecup_inference_server/sandbox/test_df_processed.csv")
    parser.add_argument('--ckpt',  default="/home/kirill/ecup_inference_server/weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    parser.add_argument('--bge_path', default='/home/kirill/ecup_inference_server/weights/models_weights/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181', help='Local path to BGEM3 model folder')
    parser.add_argument('--outdir', default='.', help='Output directory for processed CSV, embeddings PKL, and predictions')
    parser.add_argument('--embed_batch', type=int, default=64, help='Batch size for BGEM3 encoding')
    parser.add_argument('--embed_maxlen', type=int, default=512, help='Max length for BGEM3 encoding')
    parser.add_argument('--pred_batch', type=int, default=64, help='Batch size for inference')
    args = parser.parse_args()

    first_csv_path = Path(args.first_csv)
    # second_csv_path = Path(args.new_features_csv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(first_csv_path)
    import time
    print("test df shape: ", df.shape)

    if 'id' not in df.columns:
        raise ValueError('Input CSV must contain an "id" column')

    if 'description' in df.columns:
        df['description'] = df['description'].progress_apply(clean_text)
    if 'name_rus' in df.columns:
        df['name_rus'] = df['name_rus'].progress_apply(clean_text)

    processed_csv_path = outdir / "test_processed.csv"
    df.to_csv(processed_csv_path, index=False)
    start = time.time()
    print("Loading BGEM3 model from:", args.bge_path)
    bge_model = BGEM3FlagModel(args.bge_path, use_fp16=True)

    name_texts = df.get('name_rus', pd.Series([''] * len(df))).fillna('').tolist()
    desc_texts = df.get('description', pd.Series([''] * len(df))).fillna('').tolist()

    print("Encoding name_rus...")
    name_embeddings = compute_embeddings(bge_model, name_texts, batch_size=args.embed_batch, max_length=args.embed_maxlen)
    print("Encoding description...")
    desc_embeddings = compute_embeddings(bge_model, desc_texts, batch_size=args.embed_batch, max_length=args.embed_maxlen)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
