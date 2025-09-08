from typing import List

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np, pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer

from environs import TRITON_PROTOCOL, TRITON_URL, DEFAULT_BATCH, CACHE_PATH, E5_NAME

tqdm.pandas()

_e5_tok = None


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
    all_out = []
    for i in tqdm(range(0, len(texts), batch_size)):
        sl = slice(i, i + batch_size)
        chunk = texts[sl]
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

        all_out.append(vecs.astype(np.float32, copy=False))

    if not all_out:
        return np.zeros((0, 0), dtype=np.float16)

    return np.concatenate(all_out, axis=0).astype(np.float32, copy=False)


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
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def infer_probs_classifier(
    client,
    model_name: str,
    e5_input_ids: np.ndarray,        # [B, L] int64
    e5_attention_mask: np.ndarray,   # [B, L] int64
    x_categ: np.ndarray,             # [B, C] int64
    x_numer: np.ndarray,             # [B, N] fp32
    x_extra: np.ndarray,             # [B, E] fp32
    batch_size: int
) -> np.ndarray:
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