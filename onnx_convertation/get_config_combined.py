import argparse, textwrap, torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights/models_weights/8000_bert_ftt_imma_BEST.pt")
    p.add_argument("--repo_dir", default="weights/triton_repo")
    p.add_argument("--model_name", default="ecup_model")
    p.add_argument("--version", default="1")
    p.add_argument("--max_batch", type=int, default=32)
    p.add_argument("--preferred_batches", default="1,8,16,32")
    p.add_argument("--bge_dim", type=int, default=1024)  # BGE-M3 dense = 1024
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    C = len(ckpt["cat_cols"])
    N = len(ckpt["num_cols"])

    repo = Path(args.repo_dir) / args.model_name
    (repo / args.version).mkdir(parents=True, exist_ok=True)

    config = f"""
name: "{args.model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {args.max_batch}

input [
  {{ name: "e5_input_ids"            data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "e5_attention_mask"       data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "bge_name_input_ids"      data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "bge_name_attention_mask" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "bge_desc_input_ids"      data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "bge_desc_attention_mask" data_type: TYPE_INT64 dims: [-1] }},
  {{ name: "x_categ"   data_type: TYPE_INT64 dims: [{C}] }},
  {{ name: "x_numer"   data_type: TYPE_FP32  dims: [{N}] }}
]

output [
  {{ name: "probs" data_type: TYPE_FP32 dims: [22] }}
]

instance_group [ {{ kind: KIND_GPU count: 1 }} ]

dynamic_batching {{
  preferred_batch_size: [ {args.preferred_batches} ]
}}
"""
    (repo / "config.pbtxt").write_text(textwrap.dedent(config).strip())
    print("Wrote:", repo / "config.pbtxt")


if __name__ == "__main__":
    main()