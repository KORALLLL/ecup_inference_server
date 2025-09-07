import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from combined_model import CombinedClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    p.add_argument("--onnx_out", default="weights/triton_repo/ecup_model/1/final_weights.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seq_e5", type=int, default=512)
    p.add_argument("--seq_bge", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--bge_model_name", default="BAAI/bge-m3")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = CombinedClassifier(ckpt, bge_model_name=args.bge_model_name).eval()

    # Размерности табличных входов
    C = len(ckpt["cat_cols"])
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
    x_categ = torch.zeros((B, C), dtype=torch.long) if C > 0 else torch.zeros((B, 0), dtype=torch.long)
    x_numer = torch.zeros((B, N), dtype=torch.float32) if N > 0 else torch.zeros((B, 0), dtype=torch.float32)

    input_names = [
        "e5_input_ids", "e5_attention_mask",
        "bge_name_input_ids", "bge_name_attention_mask",
        "bge_desc_input_ids", "bge_desc_attention_mask",
        "x_categ", "x_numer"
    ]
    output_names = ["probs"]

    dynamic_axes = {
        "e5_input_ids": {0: "batch", 1: "seq_e5"},
        "e5_attention_mask": {0: "batch", 1: "seq_e5"},
        "bge_name_input_ids": {0: "batch", 1: "seq_bge"},
        "bge_name_attention_mask": {0: "batch", 1: "seq_bge"},
        "bge_desc_input_ids": {0: "batch", 1: "seq_bge"},
        "bge_desc_attention_mask": {0: "batch", 1: "seq_bge"},
        "x_categ": {0: "batch"},
        "x_numer": {0: "batch"},
        "probs": {0: "batch"},
    }

    Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (e5_input_ids, e5_attention_mask,
         bge_name_input_ids, bge_name_attention_mask,
         bge_desc_input_ids, bge_desc_attention_mask,
         x_categ, x_numer),
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
