import torch
import torch.nn as nn
import torch.nn.functional as F
from hyper_connections import HyperConnections
from einops import rearrange, repeat
from transformers import AutoModel


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
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir="weights/")
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

        self.bge_model = AutoModel.from_pretrained(bge_model_name, cache_dir="weights/")
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