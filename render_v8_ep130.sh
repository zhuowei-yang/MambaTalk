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

SAMPLES="Ak2Bm8mfL3w_0008_speaker2 ISgz9wLZxJI_0032_speaker2 BjhoZ0aSvPY_0002_speaker2"

mkdir -p "$OUT_DIR"

total=$(echo $SAMPLES | wc -w)
idx=0
for name in $SAMPLES; do
    idx=$((idx+1))
    echo "========================================"
    echo "[$idx/$total] $name"
    echo "========================================"

    pred_npz="${CKPT_DIR}/${EPOCH}/res_${name}.npz"
    gt_npz="${DATA_DIR}/${name}.npz"
    wav="${WAVE_DIR}/${name}.wav"

    if [ ! -f "$pred_npz" ]; then
        echo "  SKIP: $pred_npz not found"
        continue
    fi

    pred_fixed="${OUT_DIR}/pred_${name}.npz"
    python3 -c "
import numpy as np
d = dict(np.load('${pred_npz}', allow_pickle=True))
N = len(d['body_pose_params'])
if 'focal_length' in d and d['focal_length'].shape[0] == 1:
    d['focal_length'] = np.full(N, d['focal_length'][0], dtype=np.float32)
if 'scale_params' in d and d['scale_params'].ndim == 1:
    d['scale_params'] = np.tile(d['scale_params'], (N, 1))
if 'shape_params' in d and d['shape_params'].ndim == 1:
    d['shape_params'] = np.tile(d['shape_params'], (N, 1))
if 'width' in d and d['width'].shape[0] == 1:
    d['width'] = np.full(N, d['width'][0], dtype=d['width'].dtype)
if 'height' in d and d['height'].shape[0] == 1:
    d['height'] = np.full(N, d['height'][0], dtype=d['height'].dtype)
np.savez('${pred_fixed}', **d)
"

    echo "  Rendering prediction..."
    CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a $SAM_PYTHON $RENDER_SCRIPT \
        --npz_path "$pred_fixed" --mode mesh

    echo "  Rendering GT..."
    gt_copy="${OUT_DIR}/gt_${name}.npz"
    cp "$gt_npz" "$gt_copy"
    CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a $SAM_PYTHON $RENDER_SCRIPT \
        --npz_path "$gt_copy" --mode mesh

    pred_mp4="${pred_fixed%.npz}_mesh.mp4"
    gt_mp4="${gt_copy%.npz}_mesh.mp4"

    if [ -f "$pred_mp4" ] && [ -f "$gt_mp4" ]; then
        compare_silent="${OUT_DIR}/compare_${name}_silent.mp4"
        compare_mp4="${OUT_DIR}/compare_${name}.mp4"
        echo "  Creating side-by-side (Left=GT | Right=Pred)..."
        ffmpeg -y -i "$gt_mp4" -i "$pred_mp4" \
            -filter_complex "[0:v]crop=iw/2:ih:iw/2:0[gt];[1:v]crop=iw/2:ih:iw/2:0[pred];[gt][pred]hstack=inputs=2[out]" \
            -map "[out]" -c:v libx264 -crf 23 "$compare_silent" 2>/dev/null

        if [ -f "$wav" ] && [ -f "$compare_silent" ]; then
            echo "  Adding audio: $wav"
            ffmpeg -y -i "$compare_silent" -i "$wav" \
                -c:v copy -c:a aac -b:a 128k -shortest "$compare_mp4" 2>/dev/null
            rm -f "$compare_silent"
        else
            mv "$compare_silent" "$compare_mp4" 2>/dev/null
        fi

        if [ -f "$compare_mp4" ]; then
            sz=$(du -h "$compare_mp4" | cut -f1)
            echo "  -> $compare_mp4 ($sz)"
        fi

        mv "$pred_mp4" "$OUT_DIR/" 2>/dev/null
    else
        echo "  Render failed: pred=$pred_mp4 gt=$gt_mp4"
    fi
    echo ""
done

echo "========================================"
echo "Done! Videos in: $OUT_DIR/"
echo "========================================"
