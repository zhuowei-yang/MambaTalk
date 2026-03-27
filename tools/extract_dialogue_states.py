"""
Offline dialogue state extraction using SoulX-Duplug.

Processes paired speaker audio files and generates per-frame state sequences
for training DualMambaTalkV8. Bypasses the WebSocket layer to get fine-grained
5-state predictions directly from _state_predict().

Signal category filtering (per fusion_analysis.md Section 0.2):
  - All 5 dialogue state tokens are ALLOWED (Head Motion + Body Gesture domain)
  - They describe turn-taking dynamics, NOT facial expressions
  - ASR text is saved alongside for audit; facial-expression keywords are logged

Usage:
    python tools/extract_dialogue_states.py \
        --data_dir ./data_new/ \
        --pose_rep smplxflame_30 \
        --output_dir ./data_new/dialogue_states/ \
        --duplug_path /data1/yangzhuowei/new_project/concat/SoulX-Duplug_code \
        --config_path /data1/yangzhuowei/new_project/concat/SoulX-Duplug_code/config/config.yaml

State mapping (all allowed per fusion_analysis.md Section 0.2):
    0 = user_idle        -> listener posture (head_motion + body_gesture)
    1 = user_nonidle     -> active speaking gestures
    2 = user_backchannel -> nodding, small feedback gestures
    3 = user_complete    -> turn-yield posture shift
    4 = user_incomplete  -> hold/pause posture
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from loguru import logger

STATE_MAP = {
    "<|user_idle|>": 0,
    "<|user_nonidle|>": 1,
    "<|user_backchannel|>": 2,
    "<|user_complete|>": 3,
    "<|user_incomplete|>": 4,
}

STATE_SIGNAL_CATEGORY = {
    0: "head_motion+body_gesture",
    1: "head_motion+body_gesture",
    2: "head_motion+body_gesture",
    3: "head_motion+body_gesture",
    4: "head_motion+body_gesture",
}

FACIAL_KEYWORDS = {"smile", "smiling", "cry", "crying", "frown", "frowning",
                   "laugh", "laughing", "grin", "wink", "pout"}

CHUNK_SAMPLES = 2560    # 160ms @ 16kHz
BACK_SAMPLES = 15360    # 960ms context
AHEAD_SAMPLES = 640     # 40ms lookahead


def extract_states_for_audio(turn_model, audio_path, sr=16000):
    """Run SoulX-Duplug on a single audio file.

    Returns:
        states: int64 array of state ids (T_state,)
        asr_texts: list of ASR text per chunk (for audit/filtering)
    """
    audio, orig_sr = librosa.load(audio_path, sr=sr, mono=True)

    turn_model.reset()

    states = []
    asr_texts = []
    n_chunks = (len(audio) - BACK_SAMPLES - AHEAD_SAMPLES) // CHUNK_SAMPLES

    for i in range(max(1, n_chunks)):
        start = BACK_SAMPLES + i * CHUNK_SAMPLES
        chunk = audio[start:start + CHUNK_SAMPLES]
        back = audio[max(0, start - BACK_SAMPLES):start]
        ahead = audio[start + CHUNK_SAMPLES:start + CHUNK_SAMPLES + AHEAD_SAMPLES]

        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        if len(back) < BACK_SAMPLES:
            back = np.pad(back, (BACK_SAMPLES - len(back), 0))
        if len(ahead) < AHEAD_SAMPLES:
            ahead = np.pad(ahead, (0, AHEAD_SAMPLES - len(ahead)))

        try:
            state_str, delta_text, asr_buffer = turn_model.infer(chunk, back, ahead)
            state_id = STATE_MAP.get(state_str, 0)
            asr_text = delta_text if delta_text else ""
        except Exception as e:
            logger.warning(f"Inference error at chunk {i}: {e}, defaulting to idle")
            state_id = 0
            asr_text = ""

        if asr_text:
            words_lower = asr_text.lower().split()
            facial_hits = FACIAL_KEYWORDS & set(words_lower)
            if facial_hits:
                logger.debug(f"  chunk {i}: ASR contains facial keywords {facial_hits} "
                             f"(state={state_id}, text='{asr_text}') - logged for audit")

        states.append(state_id)
        asr_texts.append(asr_text)

    return np.array(states, dtype=np.int64), asr_texts


def main():
    parser = argparse.ArgumentParser(description="Extract dialogue states from audio")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--pose_rep", default="smplxflame_30", help="Pose representation subdir")
    parser.add_argument("--output_dir", required=True, help="Output directory for state files")
    parser.add_argument("--duplug_path", required=True, help="Path to SoulX-Duplug_code")
    parser.add_argument("--config_path", default=None, help="SoulX-Duplug config.yaml path")
    parser.add_argument("--split_csv", default="train_test_split.csv", help="Split CSV file")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    args = parser.parse_args()

    sys.path.insert(0, args.duplug_path)

    orig_cwd = os.getcwd()
    os.chdir(args.duplug_path)

    from service.model import TurnModel

    config_path = args.config_path or os.path.join(args.duplug_path, "config/config.yaml")
    logger.info(f"Loading SoulX-Duplug from {config_path} (cwd={os.getcwd()})")
    turn_model = TurnModel(config_path=config_path)

    os.chdir(orig_cwd)

    os.makedirs(args.output_dir, exist_ok=True)
    asr_dir = os.path.join(args.output_dir, "asr_text")
    os.makedirs(asr_dir, exist_ok=True)

    split_csv = os.path.join(args.data_dir, args.split_csv)
    split_rule = pd.read_csv(split_csv)

    speaker1_files = split_rule[split_rule['id'].str.endswith('speaker1')]
    total = len(speaker1_files)

    state_names = {v: k for k, v in STATE_MAP.items()}
    global_state_counts = np.zeros(len(STATE_MAP), dtype=np.int64)
    facial_keyword_count = 0

    for idx, (_, row) in enumerate(speaker1_files.iterrows()):
        f_name1 = row['id']
        f_name2 = f_name1.rsplit("speaker1", 1)[0] + "speaker2"

        wav1 = os.path.join(args.data_dir, "wave16k", f_name1 + ".wav")
        wav2 = os.path.join(args.data_dir, "wave16k", f_name2 + ".wav")

        out1 = os.path.join(args.output_dir, f_name1 + "_states.npy")
        out2 = os.path.join(args.output_dir, f_name2 + "_states.npy")

        if os.path.exists(out1) and os.path.exists(out2):
            logger.info(f"[{idx+1}/{total}] Skipping {f_name1} (already exists)")
            continue

        logger.info(f"[{idx+1}/{total}] Processing {f_name1} <-> {f_name2}")

        if os.path.exists(wav1):
            states1, asr1 = extract_states_for_audio(turn_model, wav1)
            np.save(out1, states1)
            with open(os.path.join(asr_dir, f_name1 + "_asr.json"), "w") as f:
                json.dump(asr1, f, ensure_ascii=False)
            for s in states1:
                global_state_counts[s] += 1
            facial_keyword_count += sum(
                1 for t in asr1 if t and (FACIAL_KEYWORDS & set(t.lower().split())))
            logger.info(f"  Speaker1: {len(states1)} states saved "
                        f"[category: {STATE_SIGNAL_CATEGORY[0]}]")
        else:
            logger.warning(f"  Speaker1 wav not found: {wav1}")

        if os.path.exists(wav2):
            states2, asr2 = extract_states_for_audio(turn_model, wav2)
            np.save(out2, states2)
            with open(os.path.join(asr_dir, f_name2 + "_asr.json"), "w") as f:
                json.dump(asr2, f, ensure_ascii=False)
            for s in states2:
                global_state_counts[s] += 1
            facial_keyword_count += sum(
                1 for t in asr2 if t and (FACIAL_KEYWORDS & set(t.lower().split())))
            logger.info(f"  Speaker2: {len(states2)} states saved "
                        f"[category: {STATE_SIGNAL_CATEGORY[0]}]")
        else:
            logger.warning(f"  Speaker2 wav not found: {wav2}")

    logger.info(f"Done. States saved to {args.output_dir}")
    logger.info("=== State Distribution ===")
    total_tokens = global_state_counts.sum()
    for sid, count in enumerate(global_state_counts):
        pct = 100.0 * count / max(total_tokens, 1)
        logger.info(f"  {state_names.get(sid, f'unknown_{sid}')}: "
                    f"{count} ({pct:.1f}%) -> {STATE_SIGNAL_CATEGORY.get(sid, 'UNKNOWN')}")
    logger.info(f"=== Facial keyword occurrences in ASR: {facial_keyword_count} "
                f"(logged for audit, not filtered from states) ===")


if __name__ == "__main__":
    main()
