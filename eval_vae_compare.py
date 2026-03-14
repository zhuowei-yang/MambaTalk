"""
Cross-project VAE reconstruction quality comparison.
Evaluates Body & Hand VAE across MambaTalk_new, MambaTalk_new_512, MambaTalk_new_512_res.
"""
import os, sys, glob
import numpy as np
import torch
from types import SimpleNamespace
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders.mhr_utils import compact_model_params_to_cont_body

DATA_DIR = "/data1/yangzhuowei/MambaTalk_new/data_new/smplxflame_30"

CONFIGS = {
    "MambaTalk_new (cb=256)": {
        "body_ckpt": "/data1/yangzhuowei/MambaTalk_new/pretrained/pretrained_vq/mhr_cont_body.bin",
        "hand_ckpt": "/data1/yangzhuowei/MambaTalk_new/pretrained/pretrained_vq/mhr_cont_hand.bin",
        "body_cb": 256, "hand_cb": 256,
        "body_class": "VQVAEConvZero", "hand_class": "VQVAEConvZero",
    },
    "MambaTalk_new_512 (cb=512/256)": {
        "body_ckpt": "/data1/yangzhuowei/MambaTalk_new_512/pretrained/pretrained_vq/mhr_cont_body.bin",
        "hand_ckpt": "/data1/yangzhuowei/MambaTalk_new_512/pretrained/pretrained_vq/mhr_cont_hand.bin",
        "body_cb": 512, "hand_cb": 256,
        "body_class": "VQVAEConvZero", "hand_class": "VQVAEConvZero",
    },
    "MambaTalk_new_512_res (cb=512/256+res)": {
        "body_ckpt": "/data1/yangzhuowei/MambaTalk_new_512_res/outputs/audio2pose/custom/0310_121753_mhr_cont_body_res/rec.bin",
        "hand_ckpt": "/data1/yangzhuowei/MambaTalk_new_512_res/outputs/audio2pose/custom/0310_121754_mhr_cont_hand_res/rec.bin",
        "body_cb": 512, "hand_cb": 256,
        "body_class": "VQVAEConvZeroRes", "hand_class": "VQVAEConvZeroRes",
    },
}

BODY_CONT_DIM = 260
HAND_DIM = 108
HAND_OFFSET = 260


def load_vae(ckpt_path, vae_test_dim, codebook_size, model_class):
    args = SimpleNamespace(
        vae_test_dim=vae_test_dim, vae_length=256,
        vae_codebook_size=codebook_size, vae_layer=2,
        vae_grow=[1, 1, 2, 1], variational=False,
        vae_quantizer_lambda=1.0, res_scale=1.0,
    )
    mod = __import__("models.motion_representation", fromlist=[model_class])
    model = getattr(mod, model_class)(args).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def get_test_samples(data_dir, max_samples=50):
    split_csv = os.path.join(os.path.dirname(data_dir), "train_test_split.csv")
    if os.path.exists(split_csv):
        import pandas as pd
        df = pd.read_csv(split_csv)
        test_ids = df.loc[df['type'] == 'test', 'id'].tolist()
        valid = [s for s in test_ids if os.path.exists(os.path.join(data_dir, s + ".npz"))]
        return valid[:max_samples]
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    return [os.path.basename(f)[:-4] for f in npz_files[:max_samples]]


def evaluate_vae(body_vae, hand_vae, samples, data_dir, has_residual=False):
    body_metrics = {"mse": [], "vel": [], "acc": [], "jitter_gt": [], "jitter_rec": []}
    hand_metrics = {"mse": [], "vel": [], "acc": [], "jitter_gt": [], "jitter_rec": []}
    body_res_norm = []

    for name in samples:
        npz_path = os.path.join(data_dir, name + ".npz")
        gt = np.load(npz_path, allow_pickle=True)

        body_euler = gt['body_pose_params'].astype(np.float32)
        hand_pca = gt['hand_pose_params'].astype(np.float32)
        n = min(body_euler.shape[0], 600)
        body_euler = body_euler[:n]
        hand_pca = hand_pca[:n]

        body_cont = compact_model_params_to_cont_body(torch.from_numpy(body_euler).float())
        hand_t = torch.from_numpy(hand_pca).float()

        with torch.no_grad():
            body_out = body_vae(body_cont.unsqueeze(0).cuda())
            hand_out = hand_vae(hand_t.unsqueeze(0).cuda())

        rec_body = body_out["rec_pose"].squeeze(0).cpu().numpy()
        rec_hand = hand_out["rec_pose"].squeeze(0).cpu().numpy()
        tar_body = body_cont.numpy()
        tar_hand = hand_pca

        for metrics, rec, tar in [(body_metrics, rec_body, tar_body), (hand_metrics, rec_hand, tar_hand)]:
            nn = min(rec.shape[0], tar.shape[0])
            r, t = rec[:nn], tar[:nn]
            metrics["mse"].append(np.mean((r - t) ** 2))
            r_vel, t_vel = np.diff(r, axis=0), np.diff(t, axis=0)
            metrics["vel"].append(np.mean(np.abs(r_vel - t_vel)))
            r_acc, t_acc = np.diff(r, n=2, axis=0), np.diff(t, n=2, axis=0)
            metrics["acc"].append(np.mean(np.abs(r_acc - t_acc)))
            metrics["jitter_gt"].append(np.mean(np.abs(t_acc)))
            metrics["jitter_rec"].append(np.mean(np.abs(r_acc)))

        if has_residual and "z_res" in body_out:
            body_res_norm.append(torch.mean(body_out["z_res"] ** 2).item())

    result = {}
    for part, metrics in [("body", body_metrics), ("hand", hand_metrics)]:
        result[f"{part}_mse"] = np.mean(metrics["mse"])
        result[f"{part}_vel"] = np.mean(metrics["vel"])
        result[f"{part}_acc"] = np.mean(metrics["acc"])
        result[f"{part}_jitter_ratio"] = np.mean(metrics["jitter_rec"]) / max(np.mean(metrics["jitter_gt"]), 1e-10)

    if body_res_norm:
        result["body_res_norm"] = np.mean(body_res_norm)

    return result


def main():
    print("Loading test samples...")
    samples = get_test_samples(DATA_DIR)
    print(f"Found {len(samples)} test samples\n")

    all_results = {}
    for proj_name, cfg in CONFIGS.items():
        print(f"Evaluating: {proj_name}")
        print(f"  Body: {cfg['body_ckpt']} (class={cfg['body_class']}, cb={cfg['body_cb']})")
        print(f"  Hand: {cfg['hand_ckpt']} (class={cfg['hand_class']}, cb={cfg['hand_cb']})")

        body_vae = load_vae(cfg["body_ckpt"], BODY_CONT_DIM, cfg["body_cb"], cfg["body_class"])
        hand_vae = load_vae(cfg["hand_ckpt"], HAND_DIM, cfg["hand_cb"], cfg["hand_class"])

        has_res = "Res" in cfg["body_class"]
        result = evaluate_vae(body_vae, hand_vae, samples, DATA_DIR, has_residual=has_res)
        all_results[proj_name] = result
        print(f"  Done.\n")

        del body_vae, hand_vae
        torch.cuda.empty_cache()

    # Body table
    body_headers = ["Project", "Body MSE", "Body Vel Err", "Body Acc Err", "Jitter Ratio"]
    body_rows = []
    for proj, r in all_results.items():
        row = [proj, f"{r['body_mse']:.6f}", f"{r['body_vel']:.6f}", f"{r['body_acc']:.6f}", f"{r['body_jitter_ratio']:.4f}"]
        if "body_res_norm" in r:
            row[0] += f" (res_norm={r['body_res_norm']:.6f})"
        body_rows.append(row)

    print("\n" + "=" * 80)
    print("Body VAE Comparison (260d continuous)")
    print("=" * 80)
    print(tabulate(body_rows, headers=body_headers, tablefmt="grid"))

    # Hand table
    hand_headers = ["Project", "Hand MSE", "Hand Vel Err", "Hand Acc Err", "Jitter Ratio"]
    hand_rows = []
    for proj, r in all_results.items():
        hand_rows.append([proj, f"{r['hand_mse']:.6f}", f"{r['hand_vel']:.6f}", f"{r['hand_acc']:.6f}", f"{r['hand_jitter_ratio']:.4f}"])

    print("\n" + "=" * 80)
    print("Hand VAE Comparison (108d PCA)")
    print("=" * 80)
    print(tabulate(hand_rows, headers=hand_headers, tablefmt="grid"))
    print()


if __name__ == "__main__":
    main()
