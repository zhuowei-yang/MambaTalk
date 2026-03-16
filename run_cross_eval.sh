#!/bin/bash
# Cross-project evaluation: MambaTalk_new vs MambaTalk_new_512 vs MambaTalk_new_512_res
# Uses evaluate_dual.py from MambaTalk project
export https_proxy=http://127.0.0.1:7893
export http_proxy=http://127.0.0.1:7893
export all_proxy=socks5://127.0.0.1:7893

GPU=${1:-4}
SAM_PYTHON="/data1/yangzhuowei/miniconda3/envs/sam_3d_body/bin/python"
EVAL_SCRIPT="/data1/yangzhuowei/MambaTalk/evaluate_dual.py"
DATA_DIR="/data1/yangzhuowei/MambaTalk_new/data_new"
GT_POSE_DIR="${DATA_DIR}/smplxflame_30"
AUDIO_DIR="${DATA_DIR}/wave16k"
LOG_DIR="/data1/yangzhuowei/MambaTalk_new_512/outputs/cross_eval"
mkdir -p "$LOG_DIR"

declare -A PROJECTS
PROJECTS["MambaTalk_new_ep90"]="/data1/yangzhuowei/MambaTalk_new/outputs/audio2pose/custom/0306_160640_mambatalk_mhr_new/90"
PROJECTS["MambaTalk_new_512_ep90"]="/data1/yangzhuowei/MambaTalk_new_512/outputs/audio2pose/custom/0310_071535_mambatalk_mhr_new/90"
PROJECTS["MambaTalk_new_512_res_ep70"]="/data1/yangzhuowei/MambaTalk_new_512_res/outputs/audio2pose/custom/0310_151413_mambatalk_mhr_new_res/70"

for name in "MambaTalk_new_ep90" "MambaTalk_new_512_ep90" "MambaTalk_new_512_res_ep70"; do
    RES_DIR="${PROJECTS[$name]}"
    EVAL_DIR="${LOG_DIR}/${name}"
    mkdir -p "$EVAL_DIR"

    echo "========================================"
    echo "Preparing: $name"
    echo "  Result dir: $RES_DIR"
    echo "========================================"

    # Generate GT npz for each res_ file
    for res_f in "$RES_DIR"/res_*.npz; do
        base=$(basename "$res_f")
        spk_id="${base#res_}"
        spk_id="${spk_id%.npz}"
        gt_src="${GT_POSE_DIR}/${spk_id}.npz"
        gt_dst="${EVAL_DIR}/gt_${spk_id}.npz"
        res_dst="${EVAL_DIR}/res_${spk_id}.npz"

        if [ -f "$gt_src" ]; then
            [ ! -f "$gt_dst" ] && cp "$gt_src" "$gt_dst"
            [ ! -f "$res_dst" ] && cp "$res_f" "$res_dst"
        fi
    done

    n_gt=$(ls "$EVAL_DIR"/gt_*.npz 2>/dev/null | wc -l)
    n_res=$(ls "$EVAL_DIR"/res_*.npz 2>/dev/null | wc -l)
    echo "  Paired: $n_gt GT, $n_res res"

    echo "  Running evaluation..."
    LOG_FILE="${LOG_DIR}/${name}.log"
    CUDA_VISIBLE_DEVICES=$GPU $SAM_PYTHON $EVAL_SCRIPT \
        --result_dir "$EVAL_DIR" \
        --data_dir "$DATA_DIR" \
        --audio_dir "$AUDIO_DIR" 2>&1 | tee "$LOG_FILE"

    echo ""
done

echo "========================================"
echo "All evaluations done! Logs in: $LOG_DIR/"
echo "========================================"
