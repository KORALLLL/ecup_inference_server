# bge_triton_gateway.py
import os
import io
import time
import pickle
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from tqdm import tqdm

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

# -------------------------
# Конфигурация окружения
# -------------------------
TRITON_URL: str = os.getenv("TRITON_URL", "http://localhost:8000")
TRITON_PROTOCOL: Literal["http", "grpc"] = os.getenv("TRITON_PROTOCOL", "http")
MODEL_NAME: str = os.getenv("TRITON_MODEL_NAME", "bge")

# Советы: max_batch_size и preferred_batch_size на стороне сервера задают динамическое батчирование [1]
DEFAULT_BATCH = int(os.getenv("EMBED_BATCH", "256"))
DEFAULT_MAXLEN = int(os.getenv("EMBED_MAXLEN", "512"))


# -------------------------
# Утилиты
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "lxml").get_text(" ")
    import re
    text = text.lower()
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_client():
    if TRITON_PROTOCOL == "grpc":
        return grpcclient.InferenceServerClient(TRITON_URL)
    return httpclient.InferenceServerClient(TRITON_URL)


def infer_texts(
    client,
    texts: List[str],
    model_name: str,
    batch_size: int = DEFAULT_BATCH
) -> np.ndarray:
    """
    Отправляет тексты на Triton модель 'bge' с входом TEXT (TYPE_STRING/BYTES) и получает dense_vecs [B, H]. [1][2]
    """
    all_out = []
    for i in tqdm(range(0, len(texts), batch_size)):
        sl = slice(i, i + batch_size)
        chunk = texts[sl]
        # Готовим массив BYTES/STRING. Для Triton используем dtype=object, элементы bytes или str [2].
        arr = np.array([t.encode("utf-8") for t in chunk], dtype=object).reshape(-1, 1)

        if TRITON_PROTOCOL == "grpc":
            inp = grpcclient.InferInput("TEXT", arr.shape, "BYTES")
            inp.set_data_from_numpy(arr)
            out = grpcclient.InferRequestedOutput("dense_vecs")
            res = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
            vecs = res.as_numpy("dense_vecs")
        else:
            inp = httpclient.InferInput("TEXT", arr.shape, "BYTES")
            inp.set_data_from_numpy(arr)
            out = httpclient.InferRequestedOutput("dense_vecs")
            res = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
            vecs = res.as_numpy("dense_vecs")

        # Triton config задаёт TYPE_FP32, получаем float32 [1]
        all_out.append(vecs.astype(np.float16, copy=False))

    if not all_out:
        return np.zeros((0, 0), dtype=np.float16)

    return np.concatenate(all_out, axis=0).astype(np.float16, copy=False)


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="BGE-M3 Triton Embeddings Gateway")


class EncodeRequest(BaseModel):
    csv_path: str
    outdir: str = "."
    batch: int = DEFAULT_BATCH
    # Не управляем max_length на клиенте, т.к. токенизация внутри сервера BGEM3FlagModel; можно проксировать через env [5]
    clean: bool = True  # применять ли clean_text


@app.post("/encode_file")
def encode_file(req: EncodeRequest):
    """
    Эндпоинт: принимает путь к CSV, очищает тексты (опционально), кодирует имени и описания отдельно на Triton 'bge',
    сохраняет PKL и возвращает путь к файлу. [1][3]
    """
    t0 = time.time()
    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return {"error": 'CSV must contain "id" column'}

    # Тексты
    name_col = "name_rus" if "name_rus" in df.columns else None
    desc_col = "description" if "description" in df.columns else None
    if name_col is None and desc_col is None:
        return {"error": 'CSV must contain at least one of ["name_rus", "description"]'}

    if req.clean and name_col:
        df[name_col] = df[name_col].fillna("").map(clean_text)
    if req.clean and desc_col:
        df[desc_col] = df[desc_col].fillna("").map(clean_text)

    name_texts = df[name_col].fillna("").astype(str).tolist() if name_col else []
    desc_texts = df[desc_col].fillna("").astype(str).tolist() if desc_col else []

    client = get_client()

    # ДВА последовательных инференса: сначала имя, затем описание (не смешиваем), как требуется [1]
    name_emb = infer_texts(client, name_texts, MODEL_NAME, batch_size=req.batch) if name_texts else np.zeros((len(df), 0), dtype=np.float32)
    desc_emb = infer_texts(client, desc_texts, MODEL_NAME, batch_size=req.batch) if desc_texts else np.zeros((len(df), 0), dtype=np.float32)

    # Сохранение в PKL в требуемом формате [ids, name_embeddings, description_embeddings]
    outdir = Path(req.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    emb_pkl_path = outdir / f"bge_embeddings_{csv_path.stem}.pkl"
    with open(emb_pkl_path, "wb") as f:
        pickle.dump(
            {
                "ids": df["id"].tolist(),
                "name_embeddings": name_emb,
                "description_embeddings": desc_emb,
            },
            f,
        )

    t1 = time.time()
    return {
        "count": int(len(df)),
        "embeddings_path": str(emb_pkl_path),
        "name_shape": tuple(name_emb.shape),
        "desc_shape": tuple(desc_emb.shape),
        "time_sec": round(t1 - t0, 3),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
