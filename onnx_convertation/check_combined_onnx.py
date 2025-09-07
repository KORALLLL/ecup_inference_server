import argparse, numpy as np, onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="weights/final_weights.onnx")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--Te5", type=int, default=128)
    p.add_argument("--Tbge", type=int, default=128)
    p.add_argument("--C", type=int, default=0)
    p.add_argument("--N", type=int, default=34)
    args = p.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)

    input_names = {i.name for i in sess.get_inputs()}
    feeds = {}
    if "e5_input_ids" in input_names:
        feeds["e5_input_ids"] = np.ones((args.B, args.Te5), np.int64)
    if "e5_attention_mask" in input_names:
        feeds["e5_attention_mask"] = np.ones((args.B, args.Te5), np.int64)
    if "bge_name_input_ids" in input_names:
        feeds["bge_name_input_ids"] = np.ones((args.B, args.Tbge), np.int64)
    if "bge_name_attention_mask" in input_names:
        feeds["bge_name_attention_mask"] = np.ones((args.B, args.Tbge), np.int64)
    if "bge_desc_input_ids" in input_names:
        feeds["bge_desc_input_ids"] = np.ones((args.B, args.Tbge), np.int64)
    if "bge_desc_attention_mask" in input_names:
        feeds["bge_desc_attention_mask"] = np.ones((args.B, args.Tbge), np.int64)
    if "x_categ" in input_names:
        feeds["x_categ"] = np.zeros((args.B, args.C), np.int64)
    if "x_numer" in input_names:
        feeds["x_numer"] = np.zeros((args.B, args.N), np.float32)

    probs = sess.run(["probs"], feeds)[0]
    print("probs:", probs.shape)
    print(probs)


if __name__ == "__main__":
    main()
