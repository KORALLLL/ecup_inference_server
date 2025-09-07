import os, io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

TRITON_URL = os.getenv("TRITON_URL", "0.0.0.0:8001")
TRITON_PROTOCOL = os.getenv("TRITON_PROTOCOL", "http")  # grpc|http
MODEL_NAME = os.getenv("MODEL_NAME", "ecup_model")
MAX_LEN_E5 = int(os.getenv("MAX_LEN_E5", "512"))
MAX_LEN_BGE = int(os.getenv("MAX_LEN_BGE", "512"))

# Два разных токенайзера
E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-small")
BGE_NAME = os.getenv("BGE_NAME", "BAAI/bge-m3")
CACHE_PATH = os.getenv("CACHE_PATH")
e5_tok = AutoTokenizer.from_pretrained(E5_NAME, cache_path=CACHE_PATH)
bge_tok = AutoTokenizer.from_pretrained(BGE_NAME, cache_path=CACHE_PATH)

# Метаданные табличных фич (загрузите ваши артефакты)
import torch
CKPT_PATH = os.getenv("WEIGHTS_PATH", "/app/checkpoints/model_ckpt.pt")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
CAT_COLS = ckpt["cat_cols"]
NUM_COLS = ckpt["num_cols"]
CAT_MAPS = ckpt["cat_maps"]
NUM_MEANS = ckpt["num_means"]; NUM_STDS = ckpt["num_stds"]


def transform_tab(X_df: pd.DataFrame):
    C = len(CAT_COLS); N = len(NUM_COLS)
    if C > 0:
        Xc = np.zeros((len(X_df), C), dtype=np.int64)
        for j, c in enumerate(CAT_COLS):
            mapping = CAT_MAPS[c]
            col_vals = X_df[c].astype(str).fillna("")
            Xc[:, j] = col_vals.map(lambda v: mapping.get(v, 0)).astype(np.int64).values
    else:
        Xc = np.zeros((len(X_df), 0), dtype=np.int64)
    if N > 0:
        Xn = np.zeros((len(X_df), N), dtype=np.float32)
        for j, c in enumerate(NUM_COLS):
            col = pd.to_numeric(X_df[c], errors="coerce").fillna(NUM_MEANS[c])
            Xn[:, j] = ((col - NUM_MEANS[c]) / NUM_STDS[c]).astype(np.float32).values
    else:
        Xn = np.zeros((len(X_df), 0), dtype=np.float32)
    return Xc, Xn


def get_client():
    return grpcclient.InferenceServerClient(TRITON_URL) if TRITON_PROTOCOL == "grpc" else httpclient.InferenceServerClient(TRITON_URL)


app = FastAPI(title="Unified E5+BGE Classifier via Triton")


@app.post("/predict")
async def predict(file: UploadFile = File(...), batch_size: int = 64, save_submission: bool = True):
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    if "id" not in df.columns:
        return {"error": 'CSV must contain "id" column'}
    for col in ["name_rus", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    def build_text_block(row):
        parts = []
        for c in ["brand_name", "name_rus", "CommercialTypeName4", "description"]:
            v = row.get(c, "")
            v = "" if pd.isna(v) else str(v)
            parts.append(f"{c}: {v}")
        return "passage: " + "\n".join(parts)

    texts_e5  = [build_text_block(r) for _, r in df.iterrows()]
    texts_name = df.get("name_rus", pd.Series([""] * len(df))).tolist()
    texts_desc = df.get("description", pd.Series([""] * len(df))).tolist()

    e5_enc = e5_tok(texts_e5, padding=True, truncation=True, max_length=MAX_LEN_E5, return_tensors="np")
    name_enc = bge_tok(texts_name, padding=True, truncation=True, max_length=MAX_LEN_BGE, return_tensors="np")
    desc_enc = bge_tok(texts_desc, padding=True, truncation=True, max_length=MAX_LEN_BGE, return_tensors="np")

    Xc, Xn = transform_tab(df)

    client = get_client()
    all_probs = []

    for i in range(0, len(df), batch_size):
        sl = slice(i, i + batch_size)
        feeds = {
            "e5_input_ids": e5_enc["input_ids"][sl].astype(np.int64),
            "e5_attention_mask": e5_enc["attention_mask"][sl].astype(np.int64),
            "bge_name_input_ids": name_enc["input_ids"][sl].astype(np.int64),
            "bge_name_attention_mask": name_enc["attention_mask"][sl].astype(np.int64),
            "bge_desc_input_ids": desc_enc["input_ids"][sl].astype(np.int64),
            "bge_desc_attention_mask": desc_enc["attention_mask"][sl].astype(np.int64),
            "x_categ": Xc[sl].astype(np.int64),
            "x_numer": Xn[sl].astype(np.float32),
        }

        if TRITON_PROTOCOL == "grpc":
            inputs = []
            for k, v in feeds.items():
                dtype = "INT64" if v.dtype == np.int64 else "FP32"
                inp = grpcclient.InferInput(k, v.shape, dtype)
                inp.set_data_from_numpy(v)
                inputs.append(inp)
            outputs = [grpcclient.InferRequestedOutput("probs")]
            res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            probs = res.as_numpy("probs")
        else:
            inputs = []
            for k, v in feeds.items():
                dtype = "INT64" if v.dtype == np.int64 else "FP32"
                inp = httpclient.InferInput(k, v.shape, dtype)
                inp.set_data_from_numpy(v)
                inputs.append(inp)
            outputs = [httpclient.InferRequestedOutput("probs")]
            res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            probs = res.as_numpy("probs")

        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0).squeeze(-1)
    thr = float(ckpt.get("best_threshold", 0.5))
    preds = (probs >= thr).astype(int)

    resp = {"count": int(len(df)), "threshold": thr, "positives": int(preds.sum())}
    if save_submission:
        subm = pd.DataFrame({"id": df["id"].astype(int).values, "prediction": preds.astype(int)})
        subm = subm.drop_duplicates("id", keep="first")
        subm.to_csv("submission.csv", index=False)
        resp["submission_path"] = "submission.csv"
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port="8001")