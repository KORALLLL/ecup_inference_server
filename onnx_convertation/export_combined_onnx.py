import os
import argparse
import torch, torch.nn as nn
from pathlib import Path

from combined_model import CombinedClassifier


class CombinedClassifierNoCateg(nn.Module):
    def __init__(self, ckpt: dict, bge_model_name):
        super().__init__()
        self.base = CombinedClassifier(ckpt, bge_model_name=bge_model_name).eval()
        self.num_categories = 0
        if getattr(self.base, "ftt", None) is not None:
            self.num_categories = int(getattr(self.base.ftt, "num_categories", 0))

    def forward(
        self,
        e5_input_ids, e5_attention_mask,
        bge_name_input_ids, bge_name_attention_mask,
        bge_desc_input_ids, bge_desc_attention_mask,
        x_numer
    ):
        B = x_numer.shape[0]
        device = x_numer.device
        if self.num_categories > 0:
            x_categ = torch.zeros((B, self.num_categories), device=device, dtype=torch.long)
        else:
            x_categ = torch.zeros((B, 0), device=device, dtype=torch.long)

        return self.base(
            e5_input_ids, e5_attention_mask,
            bge_name_input_ids, bge_name_attention_mask,
            bge_desc_input_ids, bge_desc_attention_mask,
            x_categ, x_numer
        ).unsqueeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.getenv("WEIGHTS_PATH"))
    p.add_argument("--onnx_out", default=f"{os.getenv('TRITON_REPO_PATH')}/{os.getenv('TRITON_MODEL_NAME')}/{os.getenv('TRITON_VERSION')}/model.onnx")
    p.add_argument("--opset", type=int, default=os.getenv("ONNX_UPSET"))
    p.add_argument("--seq_e5", type=int, default=int(os.getenv("MAX_LEN_E5")))
    p.add_argument("--seq_bge", type=int, default=int(os.getenv("MAX_LEN_BGE")))
    p.add_argument("--batch", type=int, default=int(os.getenv("ONNX_BATCH")))
    p.add_argument("--bge_model_name", default=os.getenv("BGE_NAME"))
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = CombinedClassifierNoCateg(ckpt, bge_model_name=args.bge_model_name).eval()

    # Размерности табличных входов
    N = len(ckpt["num_cols"])

    # Заглушки
    B = args.batch
    T_e5 = args.seq_e5
    T_bge = args.seq_bge
    e5_input_ids = torch.ones((B, T_e5), dtype=torch.long)
    e5_attention_mask = torch.ones((B, T_e5), dtype=torch.long)
    bge_name_input_ids = torch.ones((B, T_bge), dtype=torch.long)
    bge_name_attention_mask = torch.ones((B, T_bge), dtype=torch.long)
    bge_desc_input_ids = torch.ones((B, T_bge), dtype=torch.long)
    bge_desc_attention_mask = torch.ones((B, T_bge), dtype=torch.long)
    x_numer = torch.zeros((B, N), dtype=torch.float32) if N > 0 else torch.zeros((B, 0), dtype=torch.float32)

    input_names = [
        "e5_input_ids", "e5_attention_mask",
        "bge_name_input_ids", "bge_name_attention_mask",
        "bge_desc_input_ids", "bge_desc_attention_mask",
        "x_numer"
    ]
    output_names = ["probs"]

    dynamic_axes = {
        "e5_input_ids": {0: "batch", 1: "seq_e5"},
        "e5_attention_mask": {0: "batch", 1: "seq_e5"},
        "bge_name_input_ids": {0: "batch", 1: "seq_bge"},
        "bge_name_attention_mask": {0: "batch", 1: "seq_bge"},
        "bge_desc_input_ids": {0: "batch", 1: "seq_bge"},
        "bge_desc_attention_mask": {0: "batch", 1: "seq_bge"},
        "x_numer": {0: "batch"},
        "probs": {0: "batch"},
    }

    Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (e5_input_ids, e5_attention_mask,
         bge_name_input_ids, bge_name_attention_mask,
         bge_desc_input_ids, bge_desc_attention_mask,
         x_numer),
        args.onnx_out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("Exported:", args.onnx_out)


if __name__ == "__main__":
    main()
