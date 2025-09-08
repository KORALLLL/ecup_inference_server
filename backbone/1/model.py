# models/ecup_py/1/model.py
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
# import triton_python_backend_utils as pb_utils


import torch.nn as nn
from hyper_connections import HyperConnections
from einops import rearrange, repeat


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, num_residual_streams=4):
        super().__init__()
        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable=(num_residual_streams == 1)
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                init_hyper_conn(
                    dim=dim,
                    branch=Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
                ),
                init_hyper_conn(dim=dim, branch=FeedForward(dim, dropout=ff_dropout)),
            ]))

    def forward(self, x, return_attn=False):
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out, attn


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
        dim_head=16,
        dim_out=1,
        num_special_tokens=2,
        attn_dropout=0.,
        ff_dropout=0.,
        num_residual_streams=4
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout, num_residual_streams=num_residual_streams
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn=False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} category columns'
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.transformer(x, return_attn=True)
        x = x[:, 0]
        logits = self.to_logits(x)
        if not return_attn:
            return logits
        return logits, _


class E5PlusFTTClassifier(nn.Module):
    def __init__(
        self,
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
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=os.getenv("CACHE_PATH"))
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
        self.gate = nn.Sequential(nn.Linear(512 * (1 + (self.has_tabular > 0) + (self.extra_dim > 0)), 3), nn.Sigmoid())

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
            fused = (w[:, 0:1] * parts[0] + w[:, 1:2] * parts[1] + w[:, 2:3] * parts[2])
        else:
            fused = torch.stack(parts, dim=0).sum(0)

        fused = self.dropout(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits


class CombinedClassifier(nn.Module):
    def __init__(self, ckpt: dict, bge_model_name: str = "BAAI/bge-m3"):
        super().__init__()
        base = E5PlusFTTClassifier(
            model_name=ckpt["model_name"],
            cat_cardinalities=ckpt["cat_cardinalities"],
            num_continuous=len(ckpt["num_cols"]),
            tab_dim=ckpt.get("tab_hparams", {}).get("tab_dim", 128),
            tab_depth=ckpt.get("tab_hparams", {}).get("tab_depth", 8),
            tab_heads=ckpt.get("tab_hparams", {}).get("tab_heads", 8),
            tab_dim_head=ckpt.get("tab_hparams", {}).get("tab_dim_head", 16),
            tab_attn_dropout=ckpt.get("tab_hparams", {}).get("tab_attn_dropout", 0.1),
            tab_ff_dropout=ckpt.get("tab_hparams", {}).get("tab_ff_dropout", 0.1),
            tab_out_dim=ckpt.get("tab_hparams", {}).get("tab_out_dim", 256),
            dropout=ckpt.get("tab_hparams", {}).get("dropout", 0.1),
            extra_dim=ckpt.get("tab_hparams", {}).get("extra_dim", 0),
        )
        base.load_state_dict(ckpt["state_dict"], strict=True)
        base.eval()
        self.e5_encoder = base.encoder
        self.has_tabular = base.has_tabular
        self.ftt = base.ftt
        self.proj_text  = base.proj_text
        self.proj_tab   = base.proj_tab
        self.proj_extra = base.proj_extra
        self.gate       = base.gate
        self.dropout    = base.dropout
        self.classifier = base.classifier

        self.bge_model = AutoModel.from_pretrained(bge_model_name, cache_dir=os.getenv("CACHE_PATH"))
        self.bge_dim = self.bge_model.config.hidden_size

        self.extra_dim = int(base.extra_dim)
        expected_extra = self.bge_dim * 2
        if self.extra_dim != expected_extra:
            raise ValueError(f"extra_dim in ckpt={self.extra_dim} != 2*bge_dim={expected_extra}; "
                             f"обученная модель и ожидаемые BGE-эмбеддинги не совпадают по размерности.")

    def forward(
        self,
        e5_input_ids, e5_attention_mask,
        bge_name_input_ids, bge_name_attention_mask,
        bge_desc_input_ids, bge_desc_attention_mask,
        x_categ, x_numer
    ):
        e5_out = self.e5_encoder(input_ids=e5_input_ids, attention_mask=e5_attention_mask)
        text_emb = mean_pool(e5_out.last_hidden_state, e5_attention_mask)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        parts = [self.proj_text(text_emb)]

        if self.has_tabular:
            tab_emb = self.ftt(x_categ, x_numer)     # [B, tab_out_dim]
            parts.append(self.proj_tab(tab_emb))

        bge_name_out = self.bge_model(input_ids=bge_name_input_ids, attention_mask=bge_name_attention_mask)
        name_cls = bge_name_out.last_hidden_state[:, 0, :]
        name_cls = F.normalize(name_cls, p=2, dim=-1)

        bge_desc_out = self.bge_model(input_ids=bge_desc_input_ids, attention_mask=bge_desc_attention_mask)
        desc_cls = bge_desc_out.last_hidden_state[:, 0, :]
        desc_cls = F.normalize(desc_cls, p=2, dim=-1)

        x_extra = torch.cat([name_cls, desc_cls], dim=-1)  # [B, 2*bge_dim]
        if self.proj_extra is not None:
            parts.append(self.proj_extra(x_extra))
        else:
            raise RuntimeError("proj_extra is None")

        if len(parts) == 3:
            fused_raw = torch.cat(parts, dim=-1)
            w = self.gate(fused_raw)
            fused = (w[:, 0:1] * parts[0] + w[:, 1:2] * parts[1] + w[:, 2:3] * parts[2])
        else:
            fused = torch.stack(parts, dim=0).sum(0)

        fused = self.dropout(fused)
        logits = self.classifier(fused).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class TritonPythonModel:
    def initialize(self, args):
        ckpt_path = "/weights/8000_bert_ftt_imma_BEST.pt"
        bge_name  = "/weights/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.CAT_COLS = ckpt.get("cat_cols", [])
        self.NUM_COLS = ckpt.get("num_cols", [])
        self.has_categ = len(self.CAT_COLS) > 0

        self.model = CombinedClassifier(ckpt=ckpt, bge_model_name=bge_name)
        self.model.eval().to("cuda")
        torch.set_grad_enabled(False)
        # if os.getenv("USE_FP16", "1") == "1":
        #     self.model.half()

        self.expect_x_categ = self.has_categ

    def _as_torch(self, np_arr, dtype, device):
        t = torch.as_tensor(np_arr, device=device)
        return t if dtype is None else t.to(dtype)

    def execute(self, requests):
        # Собираем батч из запросов
        E5_IDS, E5_MASK, N_IDS, N_MASK, D_IDS, D_MASK, X_CAT, X_NUM = [], [], [], [], [], [], [], []
        slices = []
        total = 0
        for req in requests:
            def get(name):
                t = pb_utils.get_input_tensor_by_name(req, name)
                return None if t is None else t.as_numpy()

            e5_ids  = get("e5_input_ids");            e5_mask  = get("e5_attention_mask")
            n_ids   = get("bge_name_input_ids");      n_mask   = get("bge_name_attention_mask")
            d_ids   = get("bge_desc_input_ids");      d_mask   = get("bge_desc_attention_mask")
            x_num   = get("x_numer")
            x_cat   = get("x_categ")

            B = int(e5_ids.shape[0])
            E5_IDS.append(e5_ids); E5_MASK.append(e5_mask)
            N_IDS.append(n_ids);   N_MASK.append(n_mask)
            D_IDS.append(d_ids);   D_MASK.append(d_mask)
            X_NUM.append(x_num)

            if self.expect_x_categ:
                if x_cat is None:
                    raise pb_utils.TritonModelException("x_categ required but not provided")
                X_CAT.append(x_cat)
            else:
                X_CAT.append(np.zeros((B, 0), dtype=np.int64))

            slices.append((total, total + B))
            total += B

        # Конкатенация
        e5_ids  = np.concatenate(E5_IDS, axis=0)
        e5_mask = np.concatenate(E5_MASK, axis=0)
        n_ids   = np.concatenate(N_IDS,  axis=0)
        n_mask  = np.concatenate(N_MASK, axis=0)
        d_ids   = np.concatenate(D_IDS,  axis=0)
        d_mask  = np.concatenate(D_MASK, axis=0)
        x_num   = np.concatenate(X_NUM,  axis=0)
        x_cat   = np.concatenate(X_CAT,  axis=0)

        # Перенос на GPU
        use_fp16 = next(self.model.parameters()).dtype == torch.float16
        tfloat = torch.float16 if use_fp16 else torch.float32

        e5_ids_t  = self._as_torch(e5_ids,  torch.int64,  "cuda")
        e5_mask_t = self._as_torch(e5_mask, torch.int64,  "cuda")
        n_ids_t   = self._as_torch(n_ids,   torch.int64,  "cuda")
        n_mask_t  = self._as_torch(n_mask,  torch.int64,  "cuda")
        d_ids_t   = self._as_torch(d_ids,   torch.int64,  "cuda")
        d_mask_t  = self._as_torch(d_mask,  torch.int64,  "cuda")
        x_cat_t   = self._as_torch(x_cat,   torch.int64,  "cuda")
        x_num_t   = self._as_torch(x_num,   tfloat,       "cuda")
        
        # Инференс
        with torch.inference_mode():
            probs = self.model(
                e5_input_ids=e5_ids_t, e5_attention_mask=e5_mask_t,
                bge_name_input_ids=n_ids_t, bge_name_attention_mask=n_mask_t,
                bge_desc_input_ids=d_ids_t, bge_desc_attention_mask=d_mask_t,
                x_categ=x_cat_t, x_numer=x_num_t
            ).float().unsqueeze(-1).contiguous().cpu().numpy()

        # Возвращаем столько же ответов, сколько запросов
        responses = []
        for s, e in slices:
            out = pb_utils.Tensor("probs", probs[s:e])  # [b,1]
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        pass

if __name__ == "__main__":
    # Мини-тест CombinedClassifier на мок-данных torch.ones
    import os
    import torch
    import numpy as np
    from tqdm import tqdm

    # Параметры из окружения с дефолтами
    ckpt_path = os.getenv("CKPT_PATH", "/home/kirill/ecup_inference_server/weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    bge_name  = os.getenv("BGE_NAME", "/home/kirill/ecup_inference_server/weights/models_weights/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")

    B   = int(os.getenv("BATCH", "64"))            # размер батча
    TE5 = int(os.getenv("E5_SEQ_LEN", "512"))     # длина e5
    TN  = int(os.getenv("NAME_SEQ_LEN", "62"))    # длина name
    TD  = int(os.getenv("DESC_SEQ_LEN", "512"))   # длина desc
    use_fp16 = os.getenv("USE_FP16", "1") == "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[MAIN] Loading ckpt from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Из ckpt берём метаданные табличной части
    cat_cards = ckpt.get("cat_cardinalities", [])
    num_cols  = ckpt.get("num_cols", [])
    C = len(cat_cards)
    N = len(num_cols)

    # Инициализация модели
    print(f"[MAIN] Building model with BGE from: {bge_name}")
    model = CombinedClassifier(ckpt=ckpt, bge_model_name=bge_name).to(device).eval()
    torch.set_grad_enabled(False)
    if use_fp16:
        model.half()
        print("[MAIN] Using FP16")

    # Мок-данные: целые int64 для input_ids/mask, float для числовых
    def ones_long(shape):
        return torch.ones(shape, dtype=torch.long, device=device)
    def ones_float(shape, dtype=torch.float32):
        return torch.ones(shape, dtype=dtype, device=device)

    # Входы текста
    e5_input_ids       = ones_long((B, TE5))
    e5_attention_mask  = ones_long((B, TE5))
    bge_name_input_ids = ones_long((B, TN))
    bge_name_attn      = ones_long((B, TN))
    bge_desc_input_ids = ones_long((B, TD))
    bge_desc_attn      = ones_long((B, TD))

    # Табличные фичи
    if C > 0:
        # категориальные индексы начинаются с 0, mock=0 допустим
        x_categ = torch.zeros((B, C), dtype=torch.long, device=device)
    else:
        x_categ = torch.zeros((B, 0), dtype=torch.long, device=device)

    if use_fp16:
        x_numer = ones_float((B, N), dtype=torch.float16) if N > 0 else torch.zeros((B, 0), dtype=torch.float16, device=device)
    else:
        x_numer = ones_float((B, N), dtype=torch.float32) if N > 0 else torch.zeros((B, 0), dtype=torch.float32, device=device)

    # Прогон
    with torch.inference_mode():
        for i in tqdm(range(300)):
            probs = model(
                e5_input_ids=e5_input_ids,
                e5_attention_mask=e5_attention_mask,
                bge_name_input_ids=bge_name_input_ids,
                bge_name_attention_mask=bge_name_attn,
                bge_desc_input_ids=bge_desc_input_ids,
                bge_desc_attention_mask=bge_desc_attn,
                x_categ=x_categ,
                x_numer=x_numer
            )

    # Вывод
    probs_f32 = probs.float().detach().cpu().numpy()
    print(f"[MAIN] probs shape: {probs_f32.shape}, dtype: {probs_f32.dtype}")
    print(f"[MAIN] min={probs_f32.min():.6f}, max={probs_f32.max():.6f}, mean={probs_f32.mean():.6f}")
