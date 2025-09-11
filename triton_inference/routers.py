import time, os, pickle
from pathlib import Path

import pandas as pd, numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from tempfile import NamedTemporaryFile

from schemas import EncodeRequest, PredictRequest, RunFullRequest
from utils import clean_text, get_client, infer_texts, build_text_block, tokenize_e5_texts, infer_probs_classifier
from environs import MODEL_NAME, NUMERIC_COLS_TO_ZERO, MODEL_NAME_PRED, BEST_THRESHOLD, CHECKPOINTS_PATH


router_var = APIRouter(
    prefix="/models",
    responses={404: {"description": "Not found"}, 500: {"description": "Internal server error"}},
)


@router_var.post("/encode_file")
def encode_file(req: EncodeRequest):
    t0 = time.time()
    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return {"error": 'CSV must contain "id" column'}

    if 'description' in df.columns:
        df['description'] = df['description'].progress_apply(clean_text)
    if 'name_rus' in df.columns:
        df['name_rus'] = df['name_rus'].progress_apply(clean_text)

    processed_csv_path = "test_processed.csv"
    df.to_csv(processed_csv_path, index=False)

    name_texts = df["name_rus"].fillna("").astype(str).tolist()
    desc_texts = df["description"].fillna("").astype(str).tolist()

    client = get_client()

    name_emb = infer_texts(client, name_texts, MODEL_NAME, batch_size=req.batch) if name_texts else np.zeros((len(df), 0), dtype=np.float32)
    desc_emb = infer_texts(client, desc_texts, MODEL_NAME, batch_size=req.batch) if desc_texts else np.zeros((len(df), 0), dtype=np.float32)

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
        "processed_csv_path": str(processed_csv_path),
        "name_shape": tuple(name_emb.shape),
        "desc_shape": tuple(desc_emb.shape),
        "time_sec": round(t1 - t0, 3),
    }


@router_var.post("/predict_file")
def predict_file(req: PredictRequest):
    t0 = time.time()
    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    emb_path = Path(req.emb_pkl_path)
    if not emb_path.exists():
        return {"error": f"Embeddings PKL not found: {emb_path}"}

    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return {"error": 'CSV must contain "id" column'}

    for col in NUMERIC_COLS_TO_ZERO:
        if col in df.columns:
            df[col] = df[col].replace(np.nan, 0.0, regex=True)

    texts_e5 = [build_text_block(r) for _, r in df.iterrows()]
    e5_ids, e5_mask = tokenize_e5_texts(texts_e5, max_len=req.e5_maxlen)

    with open(emb_path, "rb") as f:
        data = pickle.load(f)
    ids_pkl = data.get("ids", [])
    name_emb = np.asarray(data.get("name_embeddings", []), dtype=np.float32)
    desc_emb = np.asarray(data.get("description_embeddings", []), dtype=np.float32)

    ckpt_path = CHECKPOINTS_PATH
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cat_cols = ckpt.get("cat_cols", [])
    num_cols = ckpt.get("num_cols", [])
    cat_maps = ckpt.get("cat_maps", {})
    num_means = ckpt.get("num_means", {})
    num_stds  = ckpt.get("num_stds", {})

    B = len(df)

    if len(cat_cols) > 0:
        x_categ = np.zeros((B, len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            mapping = cat_maps.get(c, {})
            col_vals = df.get(c, pd.Series([""] * B)).astype(str).fillna("")
            x_categ[:, j] = col_vals.map(lambda v: mapping.get(v, 0)).astype(np.int64).values
    else:
        x_categ = np.zeros((B, 0), dtype=np.int64)

    if len(num_cols) > 0:
        x_numer = np.zeros((B, len(num_cols)), dtype=np.float32)
        for j, c in enumerate(num_cols):
            col = pd.to_numeric(df.get(c, pd.Series([np.nan] * B)), errors="coerce").fillna(num_means.get(c, 0.0))
            std = num_stds.get(c, 1.0) if num_stds.get(c, 1.0) != 0 else 1.0
            x_numer[:, j] = ((col - num_means.get(c, 0.0)) / std).astype(np.float32).values
    else:
        x_numer = np.zeros((B, 0), dtype=np.float32)

    if name_emb.ndim != 2 or desc_emb.ndim != 2:
        return {"error": f"Bad embeddings rank in PKL: name={name_emb.shape}, desc={desc_emb.shape} (expect 2D arrays)"}
    x_extra = np.concatenate([name_emb, desc_emb], axis=1).astype(np.float32, copy=False)

    client = get_client()

    probs = infer_probs_classifier(
        client=client,
        model_name=MODEL_NAME_PRED,
        e5_input_ids=e5_ids,
        e5_attention_mask=e5_mask,
        x_categ=x_categ,
        x_numer=x_numer,
        x_extra=x_extra,
        batch_size=req.batch
    )

    preds = (probs >= BEST_THRESHOLD).astype(np.int32)

    hi_thr = float(os.getenv("SELLER_HI_THR", "0.8"))
    lo_thr = float(os.getenv("SELLER_LO_THR", "0.1"))

    if "SellerID" in df.columns:
        seller_pred_rate = (
            pd.DataFrame({"SellerID": df["SellerID"].values, "prediction": preds})
            .groupby("SellerID")["prediction"]
            .mean()
        )
        sellers_to_one  = seller_pred_rate[seller_pred_rate > hi_thr].index
        sellers_to_zero = seller_pred_rate[seller_pred_rate < lo_thr].index

        mask_one  = df["SellerID"].isin(sellers_to_one).to_numpy()
        mask_zero = df["SellerID"].isin(sellers_to_zero).to_numpy()

        test_pred = preds.copy()
        test_pred[mask_one]  = 1
        test_pred[mask_zero] = 0
    else:
        pass

    outdir = Path(req.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / f"preds_{csv_path.stem}.csv"
    pd.DataFrame({"id": df["id"].astype(int).values, "pred": test_pred}).to_csv(pred_path, index=False)

    t1 = time.time()
    return {
        "count": int(B),
        "preds_path": str(pred_path),
        "time_sec": round(t1 - t0, 3),
        "model": MODEL_NAME_PRED
    }


@router_var.post("/run_full")
async def run_full(
    file: UploadFile = File(...),               # CSV загружается как form-data "file"
    batch: Optional[int] = Form(None),
    clean: Optional[bool] = Form(False),
    e5_maxlen: Optional[int] = Form(None),
    outdir: Optional[str] = Form(None)
):
    # 1) Валидация типа
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expecting a .csv file")

    # 2) Сохранить во временный CSV-файл, если нижележащие функции ждут путь
    tmp_csv = NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        contents = await file.read()
        tmp_csv.write(contents)
        tmp_csv.flush()
        tmp_csv.close()
    except Exception as e:
        try:
            tmp_csv.close()
        except:
            pass
        if os.path.exists(tmp_csv.name):
            os.remove(tmp_csv.name)
        raise HTTPException(status_code=500, detail=f"Failed to read/save uploaded file: {e}")  # [2][12]

    try:
        # 3) Вызов encode
        enc_req = EncodeRequest(csv_path=tmp_csv.name, outdir=outdir, batch=batch, clean=clean)
        enc_res = encode_file(enc_req)
        if isinstance(enc_res, dict) and "error" in enc_res:
            return {"stage": "encode", **enc_res}  # JSON об ошибке на стадии encode [15]

        csv_for_pred = enc_res.get("processed_csv_path", tmp_csv.name)
        emb_pkl_path = enc_res["embeddings_path"]

        # 4) Вызов predict
        pred_req = PredictRequest(
            csv_path=csv_for_pred,
            emb_pkl_path=emb_pkl_path,
            outdir=outdir,
            batch=batch,
            e5_maxlen=e5_maxlen,
            clean=clean
        )
        pred_res = predict_file(pred_req)

        preds_path = pred_res["preds_path"]
        whole_time = (enc_res.get("time_sec") or 0) + (pred_res.get("time_sec") or 0)

        # 5) Отдать файл как скачивание
        # Если уже есть CSV-файл предиктов на диске:
        base = os.path.splitext(os.path.basename(file.filename))  # root без расширения
        download_name = f"{base[0]}_preds.csv"
        # Можно FileResponse, он сам выставит заголовки, включая Content-Disposition filename
        return FileResponse(
            path=preds_path,
            media_type="text/csv",
            filename=download_name,
            headers={"X-Whole-Time-Sec": str(whole_time)}
        )  # [15][19][9]


    finally:
        # 6) Уборка
        if os.path.exists(tmp_csv.name):
            os.remove(tmp_csv.name)  # [