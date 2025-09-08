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
import json
from typing import Optional
import torch

# -------------------------
# Конфигурация окружения
# -------------------------
TRITON_URL: str = os.getenv("TRITON_URL", "http://localhost:8000")
TRITON_PROTOCOL: Literal["http", "grpc"] = os.getenv("TRITON_PROTOCOL", "http")
MODEL_NAME: str = os.getenv("TRITON_MODEL_NAME", "bge")
MODEL_NAME_PRED: str = os.getenv("TRITON_MODEL_NAME_PRED", "backbone")

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


from transformers import AutoTokenizer


E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-small")
CACHE_PATH = os.getenv("CACHE_PATH", None)
_e5_tok = None


def get_e5_tokenizer():
    global _e5_tok
    if _e5_tok is None:
        kwargs = {}
        if CACHE_PATH:
            kwargs["cache_dir"] = CACHE_PATH
        _e5_tok = AutoTokenizer.from_pretrained(E5_NAME, **kwargs)
    return _e5_tok


def build_text_block(row: pd.Series) -> str:
    parts = []
    for c in ["brand_name", "name_rus", "CommercialTypeName4", "description"]:
        v = row.get(c, "")
        v = "" if pd.isna(v) else str(v)
        parts.append(f"{c}: {v}")
    return "passage: " + "\n".join(parts)


def tokenize_e5_texts(texts: List[str], max_len: int = 512):
    tok = get_e5_tokenizer()
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="np")
    # Возвращаем np.int64
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def infer_probs_classifier(
    client,
    model_name: str,
    e5_input_ids: np.ndarray,        # [B, L] int64
    e5_attention_mask: np.ndarray,   # [B, L] int64
    x_categ: np.ndarray,             # [B, C] int64 (может [B,0])
    x_numer: np.ndarray,             # [B, N] fp32  (может [B,0])
    x_extra: np.ndarray,             # [B, E] fp32  (может [B,0])
    batch_size: int
) -> np.ndarray:
    """
    Делает батчевый вызов второй модели (e5_ftt) и возвращает probs [B].
    В config второй модели должны совпадать имена входов/выходов.
    """
    outs = []
    total = e5_input_ids.shape[0]
    for i in tqdm(range(0, total, batch_size), desc="predict"):
        sl = slice(i, i + batch_size)
        feeds = {
            "e5_input_ids":      e5_input_ids[sl],
            "e5_attention_mask": e5_attention_mask[sl],
            "x_categ":           x_categ[sl],
            "x_numer":           x_numer[sl].astype(np.float32, copy=False),
            "x_extra":           x_extra[sl].astype(np.float32, copy=False),
        }
        if TRITON_PROTOCOL == "grpc":
            inputs = []
            for k, v in feeds.items():
                dtype = "INT64" if str(v.dtype).startswith("int") else "FP32"
                inp = grpcclient.InferInput(k, v.shape, dtype)
                inp.set_data_from_numpy(v)
                inputs.append(inp)
            out = grpcclient.InferRequestedOutput("probs")
            res = client.infer(model_name=model_name, inputs=inputs, outputs=[out])
            probs = res.as_numpy("probs")  # [b,1] fp32
        else:
            inputs = []
            for k, v in feeds.items():
                dtype = "INT64" if str(v.dtype).startswith("int") else "FP32"
                inp = httpclient.InferInput(k, v.shape, dtype)
                inp.set_data_from_numpy(v)
                inputs.append(inp)
            out = httpclient.InferRequestedOutput("probs")
            res = client.infer(model_name=model_name, inputs=inputs, outputs=[out])
            probs = res.as_numpy("probs")  # [b,1] fp32

        outs.append(probs)

    if not outs:
        return np.zeros((0,), dtype=np.float32)

    probs_full = np.concatenate(outs, axis=0).astype(np.float32, copy=False)  # [B,1]
    return probs_full.squeeze(-1)


class PredictRequest(BaseModel):
    csv_path: str
    emb_pkl_path: str
    outdir: str = "."
    batch: int = 256
    e5_maxlen: int = 512
    clean: bool = True


@app.post("/predict_file")
def predict_file(req: PredictRequest):
    """
    Эндпоинт: принимает путь к CSV и к .pkl с эмбеддингами (name_embeddings, description_embeddings),
    токенизирует тексты под E5 для классификатора, собирает x_extra из PKL, вызывает вторую модель в Triton,
    и возвращает предикты (probs и бинарные preds).
    """
    t0 = time.time()
    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    emb_path = Path(req.emb_pkl_path)
    if not emb_path.exists():
        return {"error": f"Embeddings PKL not found: {emb_path}"}

    # Читаем CSV
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return {"error": 'CSV must contain "id" column'}

    # Опциональная очистка
    for c in ["brand_name", "name_rus", "CommercialTypeName4", "description"]:
        if c in df.columns and req.clean:
            df[c] = df[c].fillna("").map(clean_text)
        elif c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Строим тексты для E5
    texts_e5 = [build_text_block(r) for _, r in df.iterrows()]
    e5_ids, e5_mask = tokenize_e5_texts(texts_e5, max_len=req.e5_maxlen)  # int64

    # Загружаем PKL эмбеддингов
    with open(emb_path, "rb") as f:
        data = pickle.load(f)
    ids_pkl = data.get("ids", [])
    name_emb = np.asarray(data.get("name_embeddings", []), dtype=np.float32)
    desc_emb = np.asarray(data.get("description_embeddings", []), dtype=np.float32)

    if len(ids_pkl) != len(df):
        # Не строго обязательно, но полезно предупредить
        pass

    # Готовим входы табличной части согласно чекпойнту
    # Загружаем чекпойнт (путь можно переопределить через переменную окружения PRED_CKPT_PATH)
    ckpt_path = os.getenv("PRED_CKPT_PATH", "/home/kirill/ecup_inference_server/weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cat_cols = ckpt.get("cat_cols", [])
    num_cols = ckpt.get("num_cols", [])
    cat_maps = ckpt.get("cat_maps", {})
    num_means = ckpt.get("num_means", {})
    num_stds  = ckpt.get("num_stds", {})

    B = len(df)

    # x_categ по mapping из чекпойнта (если есть категориальные признаки)
    if len(cat_cols) > 0:
        x_categ = np.zeros((B, len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            mapping = cat_maps.get(c, {})
            col_vals = df.get(c, pd.Series([""] * B)).astype(str).fillna("")
            x_categ[:, j] = col_vals.map(lambda v: mapping.get(v, 0)).astype(np.int64).values
    else:
        x_categ = np.zeros((B, 0), dtype=np.int64)

    # x_numer стандартизованные, как на обучении
    if len(num_cols) > 0:
        x_numer = np.zeros((B, len(num_cols)), dtype=np.float32)
        for j, c in enumerate(num_cols):
            col = pd.to_numeric(df.get(c, pd.Series([np.nan] * B)), errors="coerce").fillna(num_means.get(c, 0.0))
            std = num_stds.get(c, 1.0) if num_stds.get(c, 1.0) != 0 else 1.0
            x_numer[:, j] = ((col - num_means.get(c, 0.0)) / std).astype(np.float32).values
    else:
        x_numer = np.zeros((B, 0), dtype=np.float32)

    # x_extra = concat(name_emb, desc_emb) по оси 1
    if name_emb.ndim != 2 or desc_emb.ndim != 2:
        return {"error": f"Bad embeddings rank in PKL: name={name_emb.shape}, desc={desc_emb.shape} (expect 2D arrays)"}
    x_extra = np.concatenate([name_emb, desc_emb], axis=1).astype(np.float32, copy=False)

    # Triton клиент
    client = get_client()

    # Вызов второй модели
    probs = infer_probs_classifier(
        client=client,
        model_name=MODEL_NAME_PRED,
        e5_input_ids=e5_ids,
        e5_attention_mask=e5_mask,
        x_categ=x_categ,
        x_numer=x_numer,
        x_extra=x_extra,
        batch_size=req.batch
    )  # [B]

    # Бинарные предикты по дефолтному порогу 0.5
    preds = (probs >= 0.5).astype(np.int32)

    # Сохранение результатов
    outdir = Path(req.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / f"preds_{csv_path.stem}.csv"
    pd.DataFrame({"id": df["id"].astype(int).values, "prob": probs.astype(np.float32), "pred": preds}).to_csv(pred_path, index=False)

    t1 = time.time()
    return {
        "count": int(B),
        "preds_path": str(pred_path),
        "time_sec": round(t1 - t0, 3),
        "model": MODEL_NAME_PRED
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
