#!/bin/bash
"""
Cross-project evaluation: generate GT npz, then run evaluate_dual.py for each project.
Usage: bash run_eval_all.sh <GPU_ID>
"""
export https_proxy=http://127.0.0.1:7893
export http_proxy=http://127.0.0.1:7893
export all_proxy=socks5://127.0.0.1:7893

GPU=${1:-4}
SAM_PYTHON="/data1/yangzhuowei/miniconda3/envs/sam_3d_body/bin/python"
EVAL_SCRIPT="/data1/yangzhuowei/MambaTalk/evaluate_dual.py"
DATA_DIR="/data1/yangzhuowei/MambaTalk_new/data_new"
GT_DATA="${DATA_DIR}/smplxflame_30"
AUDIO_DIR="${DATA_DIR}/wave16k"

declare -A PROJECTS
PROJECTS["MambaTalk_new_ep90"]="/data1/yangzhuowei/MambaTalk_new/outputs/audio2pose/custom/0306_160640_mambatalk_mhr_new/90"
PROJECTS["MambaTalk_new_512_ep90"]="/data1/yangzhuowei/MambaTalk_new_512/outputs/audio2pose/custom/0310_071535_mambatalk_mhr_new/90"
PROJECTS["MambaTalk_new_512_res_ep70"]="/data1/yangzhuowei/MambaTalk_new_512_res/outputs/audio2pose/custom/0310_151413_mambatalk_mhr_new_res/70"

LOG_DIR="/data1/yangzhuowei/MambaTalk_new_512_res/outputs/eval_compare"
mkdir -p "$LOG_DIR"

for proj_name in "MambaTalk_new_ep90" "MambaTalk_new_512_ep90" "MambaTalk_new_512_res_ep70"; do
    result_dir="${PROJECTS[$proj_name]}"
    echo "========================================"
    echo "Evaluating: $proj_name"
    echo "Result dir: $result_dir"
    echo "========================================"

    # Generate GT npz files if not present
    gt_count=$(ls "$result_dir"/gt_*.npz 2>/dev/null | wc -l)
    res_count=$(ls "$result_dir"/res_*.npz 2>/dev/null | wc -l)
    echo "  Found $res_count res files, $gt_count gt files"

    if [ "$gt_count" -eq 0 ] && [ "$res_count" -gt 0 ]; then
        echo "  Generating GT npz files..."
        python3 -c "
import os, glob, numpy as np
result_dir = '${result_dir}'
gt_data = '${GT_DATA}'
res_files = sorted(glob.glob(os.path.join(result_dir, 'res_*.npz')))
count = 0
for rf in res_files:
    name = os.path.basename(rf).replace('res_', '')
    speaker_id = name.replace('.npz', '')
    gt_src = os.path.join(gt_data, name)
    gt_dst = os.path.join(result_dir, 'gt_' + name)
    if os.path.exists(gt_src) and not os.path.exists(gt_dst):
        gt = np.load(gt_src, allow_pickle=True)
        N_res = np.load(rf, allow_pickle=True)['body_pose_params'].shape[0]
        d = {}
        d['body_pose_params'] = gt['body_pose_params'][:N_res].astype(np.float32)
        d['hand_pose_params'] = gt['hand_pose_params'][:N_res].astype(np.float32)
        d['global_rot'] = gt['global_rot'][:N_res].astype(np.float32)
        d['expr_params'] = gt['expr_params'][:N_res].astype(np.float32) if 'expr_params' in gt.files else np.zeros((N_res,72),dtype=np.float32)
        d['pred_cam_t'] = gt['pred_cam_t'][:N_res].astype(np.float32) if 'pred_cam_t' in gt.files else np.zeros((N_res,3),dtype=np.float32)
        sp = gt['shape_params']
        d['shape_params'] = sp if sp.ndim==1 else sp[0]
        sc = gt['scale_params']
        d['scale_params'] = sc if sc.ndim==1 else sc[0]
        np.savez(gt_dst, **d)
        count += 1
print(f'Generated {count} GT files')
"
    fi

    # Run evaluation
    log_file="${LOG_DIR}/${proj_name}.log"
    echo "  Running evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU $SAM_PYTHON $EVAL_SCRIPT \
        --result_dir "$result_dir" \
        --data_dir "$DATA_DIR" \
        --audio_dir "$AUDIO_DIR" \
        2>&1 | tee "$log_file"

    echo ""
done

echo "========================================"
echo "All evaluations done! Logs in: $LOG_DIR/"
echo "========================================"
