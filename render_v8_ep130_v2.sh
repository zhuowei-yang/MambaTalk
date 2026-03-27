#!/bin/bash
export PYOPENGL_PLATFORM=osmesa

GPU=${1:-4}
EPOCH=130
BASE_DIR="/data1/yangzhuowei/MambaTalk_new_512_proj_global_v8"
CKPT_DIR="${BASE_DIR}/outputs/audio2pose/custom/0321_060512_mambatalk_mhr_new"
DATA_DIR="${BASE_DIR}/data_new/smplxflame_30"
WAVE_DIR="${BASE_DIR}/data_new/wave16k"
OUT_DIR="${BASE_DIR}/outputs/render_test_v8_ep${EPOCH}"
RENDER_SCRIPT="/data1/yangzhuowei/Gesture/data_process/render_mesh_video_from_npz.py"
SAM_PYTHON="/data1/yangzhuowei/miniconda3/envs/sam_3d_body/bin/python"

SAMPLES="VWeRsVnYMMQ_0010_speaker1 ISgz9wLZxJI_0032_speaker1 F3-79a9H9Zg_0005_speaker1"

mkdir -p "$OUT_DIR"

total=$(echo $SAMPLES | wc -w)
idx=0
for name in $SAMPLES; do
    idx=$((idx+1))
    compare_mp4="${OUT_DIR}/compare_${name}.mp4"
    if [ -f "$compare_mp4" ]; then
        echo "[SKIP] $name already rendered"
        continue
    fi
    echo "========================================"
    echo "[$idx/$total] $name"
    echo "========================================"

    pred_npz="${CKPT_DIR}/${EPOCH}/res_${name}.npz"
    gt_npz="${DATA_DIR}/${name}.npz"
    wav="${WAVE_DIR}/${name}.wav"

    if [ ! -f "$pred_npz" ]; then echo "  SKIP: pred not found"; continue; fi

    pred_fixed="${OUT_DIR}/pred_${name}.npz"
    gt_copy="${OUT_DIR}/gt_${name}.npz"
    python3 -c "
import numpy as np

def fix_npz(src, dst):
    d = dict(np.load(src, allow_pickle=True))
    N = len(d['body_pose_params'])
    out = {}
    skip_keys = {'frame_idx'}
    for k, v in d.items():
        if k in skip_keys:
            continue
        if hasattr(v, 'shape') and len(v.shape) >= 1:
            if v.shape[0] == N:
                out[k] = v
            elif v.shape[0] == 1:
                out[k] = np.tile(v, (N,) + (1,)*(len(v.shape)-1))
            elif v.shape[0] < N:
                pad = np.zeros((N - v.shape[0],) + v.shape[1:], dtype=v.dtype)
                out[k] = np.concatenate([v, pad], axis=0)
            else:
                out[k] = v[:N]
        else:
            out[k] = v
    np.savez(dst, **out)
    return N

N_pred = fix_npz('${pred_npz}', '${pred_fixed}')
N_gt = fix_npz('${gt_npz}', '${gt_copy}')
print(f'Frames: pred={N_pred}, gt={N_gt}')
"

    echo "  Rendering prediction..."
    CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a $SAM_PYTHON $RENDER_SCRIPT \
        --npz_path "$pred_fixed" --mode mesh

    echo "  Rendering GT..."
    CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a $SAM_PYTHON $RENDER_SCRIPT \
        --npz_path "$gt_copy" --mode mesh

    pred_mp4="${pred_fixed%.npz}_mesh.mp4"
    gt_mp4="${gt_copy%.npz}_mesh.mp4"

    if [ -f "$pred_mp4" ] && [ -f "$gt_mp4" ]; then
        compare_silent="${OUT_DIR}/compare_${name}_silent.mp4"
        echo "  Creating side-by-side (Left=GT | Right=Pred)..."
        ffmpeg -y -i "$gt_mp4" -i "$pred_mp4" \
            -filter_complex "[0:v]crop=iw/2:ih:iw/2:0[gt];[1:v]crop=iw/2:ih:iw/2:0[pred];[gt][pred]hstack=inputs=2[out]" \
            -map "[out]" -c:v libx264 -crf 23 "$compare_silent" 2>/dev/null

        if [ -f "$wav" ] && [ -f "$compare_silent" ]; then
            echo "  Adding audio..."
            ffmpeg -y -i "$compare_silent" -i "$wav" \
                -c:v copy -c:a aac -b:a 128k -shortest "$compare_mp4" 2>/dev/null
            rm -f "$compare_silent"
        else
            mv "$compare_silent" "$compare_mp4" 2>/dev/null
        fi
        [ -f "$compare_mp4" ] && echo "  -> $compare_mp4 ($(du -h "$compare_mp4" | cut -f1))"
    else
        echo "  Render failed!"
        [ ! -f "$pred_mp4" ] && echo "    pred mp4 missing"
        [ ! -f "$gt_mp4" ] && echo "    gt mp4 missing"
    fi
    echo ""
done

echo "========================================"
echo "Done! All compare videos:"
ls -lh "$OUT_DIR"/compare_*.mp4 2>/dev/null
echo "========================================"
