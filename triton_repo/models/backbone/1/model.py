import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import triton_python_backend_utils as pb_utils

from transformers import AutoTokenizer, AutoModel
from einops import rearrange, repeat
from hyper_connections import HyperConnections


MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LEN_TEXT = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_COLS = ["brand_name", "name_rus", "description", "CommercialTypeName4"]
E5_PREFIX = "passage: "

NUMERIC_COLS_TO_ZERO = [
    'rating_1_count','rating_2_count','rating_3_count','rating_4_count','rating_5_count',
    'comments_published_count','photos_published_count','videos_published_count',
    'ExemplarAcceptedCountTotal7','ExemplarAcceptedCountTotal30','ExemplarAcceptedCountTotal90',
    'OrderAcceptedCountTotal7','OrderAcceptedCountTotal30','OrderAcceptedCountTotal90',
    'ExemplarReturnedCountTotal7','ExemplarReturnedCountTotal30','ExemplarReturnedCountTotal90',
    'ExemplarReturnedValueTotal7','ExemplarReturnedValueTotal30','ExemplarReturnedValueTotal90',
    'ItemVarietyCount','ItemAvailableCount',
    'GmvTotal7','GmvTotal30','GmvTotal90',
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "lxml").get_text(" ")
    text = text.lower()
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_text_block(row: pd.Series, cols) -> str:
    ordered = ["brand_name", "name_rus", "CommercialTypeName4", "description"]
    parts = []
    for c in ordered:
        if c in cols:
            v = row.get(c, "")
            v = "" if pd.isna(v) else str(v)
            parts.append(f"{c}: {v}")
    return E5_PREFIX + "\n".join(parts)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, num_residual_streams = 4):
        super().__init__()
        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable = num_residual_streams == 1
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []
        x = self.expand_streams(x)
        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = ff(x)
        x = self.reduce_streams(x)
        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))
    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_residual_streams = 4
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim = dim, depth = depth, heads = heads, dim_head = dim_head,
            attn_dropout = attn_dropout, ff_dropout = ff_dropout, num_residual_streams = num_residual_streams
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} category columns'
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim = 1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x, _ = self.transformer(x, return_attn = True)
        x = x[:, 0]
        logits = self.to_logits(x)
        if not return_attn:
            return logits
        return logits, _


class E5PlusFTTClassifier(nn.Module):
    def __init__(self,
        model_name: str,
        cat_cardinalities,
        num_continuous: int,
        tab_dim: int = 64,
        tab_depth: int = 4,
        tab_heads: int = 8,
        tab_dim_head: int = 16,
        tab_attn_dropout: float = 0.1,
        tab_ff_dropout: float = 0.1,
        tab_out_dim: int = 128,
        dropout: float = 0.1,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        text_hidden = self.encoder.config.hidden_size

        self.has_tabular = (len(cat_cardinalities) + num_continuous) > 0
        if self.has_tabular:
            self.ftt = FTTransformer(
                categories=tuple(cat_cardinalities),
                num_continuous=num_continuous,
                dim=tab_dim, depth=tab_depth, heads=tab_heads, dim_head=tab_dim_head,
                dim_out=tab_out_dim, attn_dropout=tab_attn_dropout, ff_dropout=tab_ff_dropout
            )
            tab_out = tab_out_dim
        else:
            self.ftt = None
            tab_out = 0

        self.extra_dim = int(extra_dim)

        self.proj_text  = nn.Sequential(nn.LayerNorm(text_hidden), nn.Linear(text_hidden, 512))
        self.proj_tab   = nn.Sequential(nn.LayerNorm(tab_out),     nn.Linear(tab_out,    512)) if self.has_tabular else None
        self.proj_extra = nn.Sequential(nn.LayerNorm(self.extra_dim), nn.Linear(self.extra_dim, 512)) if self.extra_dim > 0 else None
        self.gate = nn.Sequential(nn.Linear(512 * (1 + (self.has_tabular>0) + (self.extra_dim>0)), 3), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self._tab_hparams = dict(
            tab_dim=tab_dim, tab_depth=tab_depth, tab_heads=tab_heads, tab_dim_head=tab_dim_head,
            tab_attn_dropout=tab_attn_dropout, tab_ff_dropout=tab_ff_dropout, tab_out_dim=tab_out_dim, dropout=dropout,
            extra_dim=self.extra_dim,
        )

    def forward(self, batch_enc, x_categ: torch.Tensor, x_numer: torch.Tensor, x_extra: torch.Tensor):
        out = self.encoder(**batch_enc)
        text_emb = mean_pool(out.last_hidden_state, batch_enc["attention_mask"])
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        parts = [self.proj_text(text_emb)]

        if self.has_tabular:
            tab_emb = self.ftt(x_categ, x_numer)
            parts.append(self.proj_tab(tab_emb))
        if self.extra_dim > 0:
            parts.append(self.proj_extra(x_extra))

        fused = torch.cat(parts, dim=-1)
        if len(parts) == 3:
            w = self.gate(fused)
            fused = (w[:,0:1]*parts[0] + w[:,1:2]*parts[1] + w[:,2:3]*parts[2])
        else:
            fused = torch.stack(parts, dim=0).sum(0)

        fused = self.dropout(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits

class MultiE5TabDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y, X_categ: np.ndarray, X_numer: np.ndarray, X_extra: np.ndarray):
        self.X = X.reset_index(drop=True)
        self.y = None
        self.Xc = X_categ
        self.Xn = X_numer
        self.Xe = X_extra
        for c in TEXT_COLS:
            if c in self.X.columns:
                self.X[c] = self.X[c].fillna("")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        text = to_text_block(row, TEXT_COLS)
        return {
            "text": text,
            "x_categ": self.Xc[idx],
            "x_numer": self.Xn[idx],
            "x_extra": self.Xe[idx],
        }


class MultiCollator:
    def __init__(self, tokenizer, max_len_text: int = MAX_LEN_TEXT):
        self.tokenizer = tokenizer
        self.max_len_text = max_len_text
    def __call__(self, batch):
        texts  = [b["text"] for b in batch]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len_text, return_tensors="pt")
        x_categ = torch.tensor(np.stack([b["x_categ"] for b in batch], axis=0), dtype=torch.long)
        x_numer = torch.tensor(np.stack([b["x_numer"] for b in batch], axis=0), dtype=torch.float)
        x_extra = torch.tensor(np.stack([b["x_extra"] for b in batch], axis=0), dtype=torch.float)
        return enc, x_categ, x_numer, x_extra


def transform_tab(X_df: pd.DataFrame, cat_cols, num_cols, cat_maps, num_means, num_stds):
    if len(cat_cols) > 0:
        X_categ = np.zeros((len(X_df), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            mapping = cat_maps[c]
            col_vals = X_df[c].astype(str).fillna("")
            X_categ[:, j] = col_vals.map(lambda v: mapping.get(v, 0)).astype(np.int64).values
    else:
        X_categ = np.zeros((len(X_df), 0), dtype=np.int64)

    if len(num_cols) > 0:
        X_numer = np.zeros((len(X_df), len(num_cols)), dtype=np.float32)
        for j, c in enumerate(num_cols):
            col = pd.to_numeric(X_df[c], errors="coerce").fillna(num_means[c])
            X_numer[:, j] = ((col - num_means[c]) / num_stds[c]).astype(np.float32).values
    else:
        X_numer = np.zeros((len(X_df), 0), dtype=np.float32)

    return X_categ, X_numer


def _to_str_ids(arr):
    return [str(x) for x in arr]


def load_extra_store(pkl_path: str, keys):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    ids = _to_str_ids(data["ids"])
    stacks = []
    for k in keys:
        arr = np.asarray(data[k], dtype=np.float32)
        stacks.append(arr)
    full = np.concatenate(stacks, axis=1).astype(np.float32)
    extra_dim = full.shape[1]
    id_to_vec = {i: v for i, v in zip(ids, full)}
    return id_to_vec, extra_dim


def build_extra_matrix(df: pd.DataFrame, id_col: str, id_to_vec, extra_dim: int) -> np.ndarray:
    Z = np.zeros((len(df), extra_dim), dtype=np.float32)
    sid = _to_str_ids(df[id_col].tolist())
    for i, k in enumerate(sid):
        v = id_to_vec.get(k)
        if v is not None:
            Z[i] = v
    return Z


def load_model_from_ckpt(ckpt: dict) -> E5PlusFTTClassifier:
    tab_hp = ckpt.get("tab_hparams", {})
    model = E5PlusFTTClassifier(
        model_name=ckpt["model_name"],
        cat_cardinalities=ckpt["cat_cardinalities"],
        num_continuous=len(ckpt["num_cols"]),
        tab_dim=tab_hp.get("tab_dim", 128),
        tab_depth=tab_hp.get("tab_depth", 8),
        tab_heads=tab_hp.get("tab_heads", 8),
        tab_dim_head=tab_hp.get("tab_dim_head", 16),
        tab_attn_dropout=tab_hp.get("tab_attn_dropout", 0.1),
        tab_ff_dropout=tab_hp.get("tab_ff_dropout", 0.1),
        tab_out_dim=tab_hp.get("tab_out_dim", 256),
        dropout=tab_hp.get("dropout", 0.1),
        extra_dim=tab_hp.get("extra_dim", 0),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


@torch.no_grad()
def predict_from_checkpoint(ckpt_path: str, df: pd.DataFrame, batch_size: int = 64, device: str = DEVICE, extra_embeddings_path: str = None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cat_cols = ckpt["cat_cols"]
    num_cols = ckpt["num_cols"]
    cat_maps = ckpt["cat_maps"]
    num_means = ckpt["num_means"]
    num_stds = ckpt["num_stds"]
    cat_cardinalities = ckpt["cat_cardinalities"]
    best_thr = float(ckpt.get("best_threshold", 0.5))

    extra_keys = ckpt.get("extra_emb_keys", ["name_embeddings", "description_embeddings"])  # default
    id_key = ckpt.get("id_key", "id")

    if extra_embeddings_path is None:
        raise ValueError("extra_embeddings_path is required for inference")

    id_to_vec_store, extra_dim = load_extra_store(extra_embeddings_path, extra_keys)

    model = load_model_from_ckpt(ckpt).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"]) 

    X_df = df.copy()
    for c in TEXT_COLS:
        if c in X_df.columns:
            X_df[c] = X_df[c].fillna("")

    Xc, Xn = transform_tab(X_df, cat_cols, num_cols, cat_maps, num_means, num_stds)
    Xe     = build_extra_matrix(X_df, id_key, id_to_vec_store, extra_dim)

    ds = MultiE5TabDataset(X_df, y=None, X_categ=Xc, X_numer=Xn, X_extra=Xe)
    collator = MultiCollator(tokenizer, max_len_text=MAX_LEN_TEXT)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    all_probs = []
    for enc, x_categ, x_numer, x_extra in tqdm(loader, desc="predict", leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        x_categ = x_categ.to(device)
        x_numer = x_numer.to(device)
        x_extra = x_extra.to(device)
        logits = model(enc, x_categ, x_numer, x_extra)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs)
    preds = (probs >= best_thr).astype(int)
    return probs, preds, best_thr

class TritonPythonModel:
    def initialize(self, args):
        # Читаем конфиг модели (для типов выходов)
        self.model_config = json.loads(args["model_config"])
        # Пути и флаги из окружения
        ckpt_path = os.getenv("CKPT_PATH", "/weights/8000_bert_ftt_imma_BEST.pt")
        use_fp16  = os.getenv("USE_FP16", "0") == "1"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загружаем чекпойнт
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.cat_cols = ckpt.get("cat_cols", [])
        self.num_cols = ckpt.get("num_cols", [])
        self.cat_maps = ckpt.get("cat_maps", {})
        self.num_means = ckpt.get("num_means", {})
        self.num_stds  = ckpt.get("num_stds", {})
        cat_cards = ckpt.get("cat_cardinalities", [])
        tab_hp = ckpt.get("tab_hparams", {})
        extra_dim = int(tab_hp.get("extra_dim", 0))

        # Строим модель и грузим веса
        self.model = E5PlusFTTClassifier(
            model_name=ckpt["model_name"],
            cat_cardinalities=cat_cards,
            num_continuous=len(self.num_cols),
            tab_dim=tab_hp.get("tab_dim", 128),
            tab_depth=tab_hp.get("tab_depth", 8),
            tab_heads=tab_hp.get("tab_heads", 8),
            tab_dim_head=tab_hp.get("tab_dim_head", 16),
            tab_attn_dropout=tab_hp.get("tab_attn_dropout", 0.1),
            tab_ff_dropout=tab_hp.get("tab_ff_dropout", 0.1),
            tab_out_dim=tab_hp.get("tab_out_dim", 256),
            dropout=tab_hp.get("dropout", 0.1),
            extra_dim=extra_dim,
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()
        torch.set_grad_enabled(False)
        if use_fp16 and device == "cuda":
            self.model.half()

        self.device = device
        self.use_fp16 = use_fp16

    def _as_torch(self, np_arr, dtype, device):
        t = torch.as_tensor(np_arr, device=device)
        return t if dtype is None else t.to(dtype)

    def execute(self, requests):
        # Сбор батча из запросов
        E5_IDS, E5_MASK, X_CAT, X_NUM, X_EXTRA = [], [], [], [], []
        slices = []
        total = 0
        for req in requests:
            def get(name):
                ten = pb_utils.get_input_tensor_by_name(req, name)
                return None if ten is None else ten.as_numpy()

            e5_ids  = get("e5_input_ids")
            e5_mask = get("e5_attention_mask")
            x_cat   = get("x_categ")
            x_num   = get("x_numer")
            x_extra = get("x_extra")  # можно отправлять нулевую форму [B,0] если не используется

            if e5_ids is None or e5_mask is None or x_num is None or x_cat is None or x_extra is None:
                return [pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Missing one of required inputs: e5_input_ids, e5_attention_mask, x_categ, x_numer, x_extra"))
                ]

            B = int(e5_ids.shape[0])
            E5_IDS.append(e5_ids); E5_MASK.append(e5_mask)
            X_CAT.append(x_cat);   X_NUM.append(x_num); X_EXTRA.append(x_extra)
            slices.append((total, total + B))
            total += B

        # Конкатенация
        e5_ids  = np.concatenate(E5_IDS, axis=0)
        e5_mask = np.concatenate(E5_MASK, axis=0)
        x_cat   = np.concatenate(X_CAT, axis=0)
        x_num   = np.concatenate(X_NUM, axis=0)
        x_extra = np.concatenate(X_EXTRA, axis=0)

        # Типы
        use_fp16 = self.use_fp16 and (self.device == "cuda")
        tfloat = torch.float16 if use_fp16 else torch.float32

        # На устройство
        e5_ids_t  = self._as_torch(e5_ids,  torch.int64, self.device)
        e5_mask_t = self._as_torch(e5_mask, torch.int64, self.device)
        x_cat_t   = self._as_torch(x_cat,   torch.int64, self.device)
        x_num_t   = self._as_torch(x_num,   tfloat,      self.device)
        x_extra_t = self._as_torch(x_extra, tfloat,      self.device)

        # Инференс
        with torch.inference_mode():
            enc = {"input_ids": e5_ids_t, "attention_mask": e5_mask_t}
            logits = self.model(enc, x_cat_t, x_num_t, x_extra_t)
            probs  = torch.sigmoid(logits).float().unsqueeze(-1).contiguous().cpu().numpy()

        # Ответы по слайсам
        responses = []
        for s, e in slices:
            out = pb_utils.Tensor("probs", probs[s:e])  # [b, 1]
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        pass

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV, compute BGEM3 embeddings, and run inference with E5+FTT model")
    parser.add_argument('--first_csv', default="/home/kirill/ecup_inference_server/sandbox/test_df_processed.csv")
    parser.add_argument('--ckpt',  default="/home/kirill/ecup_inference_server/weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    parser.add_argument('--outdir', default='.', help='Output directory for processed CSV, embeddings PKL, and predictions')
    parser.add_argument('--pred_batch', type=int, default=64, help='Batch size for inference')
    args = parser.parse_args()
    emb_pkl_path = "/home/kirill/ecup_inference_server/sandbox/test_embeddings.pkl"

    first_csv_path = Path(args.first_csv)
    # second_csv_path = Path(args.new_features_csv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(first_csv_path)
    import time
    print("test df shape: ", df.shape)

    if 'id' not in df.columns:
        raise ValueError('Input CSV must contain an "id" column')

    for col in NUMERIC_COLS_TO_ZERO:
        if col in df.columns:
            df[col] = df[col].replace(np.nan, 0.0, regex=True)

    if 'description' in df.columns:
        df['description'] = df['description'].progress_apply(clean_text)
    if 'name_rus' in df.columns:
        df['name_rus'] = df['name_rus'].progress_apply(clean_text)

    processed_csv_path = outdir / f"test_processed.csv"
    df.to_csv(processed_csv_path, index=False)
    start = time.time()

    print("Loading checkpoint and running inference...")
    probs, preds, thr = predict_from_checkpoint(
        ckpt_path=args.ckpt,
        df=df,
        batch_size=args.pred_batch,
        device=DEVICE,
        extra_embeddings_path=str(emb_pkl_path),
    )

    t = 0.63
    test_pred = (probs >= t).astype(int)
    print(sum(test_pred))

    hi_thr = 0.8
    lo_thr = 0.1

    seller_pred_rate = (
        pd.DataFrame({
            "SellerID": df["SellerID"].values,
            "prediction": test_pred.astype(int)
        })
        .groupby("SellerID")["prediction"]
        .mean()
    )

    sellers_to_set_all_one = seller_pred_rate[seller_pred_rate > hi_thr].index
    mask_one = df["SellerID"].isin(sellers_to_set_all_one)

    sellers_to_set_all_zero = seller_pred_rate[seller_pred_rate < lo_thr].index
    mask_zero = df["SellerID"].isin(sellers_to_set_all_zero)

    test_pred = test_pred.copy()

    test_pred[mask_one.to_numpy()] = 1
    test_pred[mask_zero.to_numpy()] = 0

    end = time.time()
    print(end-start)


if __name__ == '__main__':
    main()
