from pydantic import BaseModel

from environs import DEFAULT_BATCH, DEFAULT_MAXLEN


class EncodeRequest(BaseModel):
    csv_path: str
    outdir: str = "."
    batch: int = DEFAULT_BATCH
    clean: bool = True


class PredictRequest(BaseModel):
    csv_path: str
    emb_pkl_path: str
    outdir: str = "."
    batch: int = DEFAULT_BATCH
    e5_maxlen: int = DEFAULT_MAXLEN
    clean: bool = True


class RunFullRequest(BaseModel):
    csv_path: str
    outdir: str = "."
    batch: int = 256
    e5_maxlen: int = 512
    clean: bool = True