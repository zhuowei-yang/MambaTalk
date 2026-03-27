"""
Dual-person MHR DataLoader for DualMambaTalkV8.
Loads paired speaker1/speaker2 MHR npz data with optional pre-computed
dialogue state sequences from SoulX-Duplug.

Cache layout per sample:
  pose1(381) + audio1 + word1 + vid1 + trans1 + shape1 + jvel1 +
  pose2(381) + audio2 + word2 + vid2 + trans2 + shape2 + jvel2 +
  state1 + state2
"""
import os
import pickle
import math
import shutil
import numpy as np
import lmdb
import textgrid as tg
import pandas as pd
import torch
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
import librosa

from .build_vocab import Vocab
from .mhr_utils import compact_model_params_to_cont_body, euler3_to_rot6d
from .beat_mhr import (
    load_mhr_native, BODY_CONT_DIM, HAND_DIM, GLOBAL_ROT6D_DIM,
    CAM_T_DIM, CONTACT_DIM, TOTAL_DIM, SELECTED_JOINT_IDXS, JOINT_VEL_DIM,
)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        self.rank = dist.get_rank()
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0, 0]
        self.joints = 55

        if hasattr(args, 'audio_rep') and args.audio_rep == "amplitude+ctc+audio":
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        split_rule = pd.read_csv(args.data_path + "train_test_split.csv")
        selected = split_rule.loc[split_rule['type'] == loader_type]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[split_rule['type'] == 'additional']
            selected = pd.concat([selected, split_b])

        self.selected_file = selected[selected['id'].str.endswith('speaker1')]
        if self.selected_file.empty:
            logger.warning(f"{loader_type} empty after speaker1 filter, fallback to train 0-8")
            fallback = split_rule.loc[split_rule['type'] == 'train']
            self.selected_file = fallback[fallback['id'].str.endswith('speaker1')].iloc[0:8]

        self.data_dir = args.data_path

        if loader_type == "test":
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr

        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)

        self.state_dir = getattr(args, 'state_dir', None)

        preloaded_dir = (self.args.root_path + self.args.cache_path
                         + loader_type + f"/{args.pose_rep}_dual_cache")
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

        if self.loader_type == "test" and len(self.selected_file) != self.n_samples:
            self._align_selected_file_with_cache()

    def _align_selected_file_with_cache(self):
        """Drop speaker1 entries from selected_file whose pair data is missing,
        so that selected_file indices align 1:1 with LMDB cache keys."""
        drop_idx = []
        for index, row in self.selected_file.iterrows():
            f1 = row['id']
            f2 = f1.rsplit("speaker1", 1)[0] + "speaker2"
            npz1_ok = os.path.exists(self.data_dir + self.args.pose_rep + "/" + f1 + ".npz")
            npz2_ok = os.path.exists(self.data_dir + self.args.pose_rep + "/" + f2 + ".npz")
            wav1_ok = self.args.audio_rep is None or os.path.exists(
                self.data_dir + "wave16k/" + f1 + ".wav")
            wav2_ok = self.args.audio_rep is None or os.path.exists(
                self.data_dir + "wave16k/" + f2 + ".wav")
            tg1_ok = self.args.word_rep is None or os.path.exists(
                self.data_dir + self.args.word_rep + "/" + f1 + ".TextGrid")
            tg2_ok = self.args.word_rep is None or os.path.exists(
                self.data_dir + self.args.word_rep + "/" + f2 + ".TextGrid")
            if not (npz1_ok and npz2_ok and wav1_ok and wav2_ok and tg1_ok and tg2_ok):
                drop_idx.append(index)
        if drop_idx:
            self.selected_file = self.selected_file.drop(drop_idx).reset_index(drop=True)
            logger.warning(
                f"Aligned dual selected_file with cache: dropped {len(drop_idx)} missing pairs, "
                f"now {len(self.selected_file)} (cache has {self.n_samples})")

    def __len__(self):
        return self.n_samples

    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading dual data '{}'...".format(self.data_dir))
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the dual cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(preloaded_dir, True, 0, 0, is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering,
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)

    def _load_speaker(self, f_name):
        """Load and preprocess data for a single speaker. Returns dict or None."""
        pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ".npz"
        if not os.path.exists(pose_file):
            return None

        mhr = load_mhr_native(pose_file)
        assert 30 % self.args.pose_fps == 0
        stride = int(30 / self.args.pose_fps)

        body_euler = mhr['body_pose'][::stride]
        hand_pca = mhr['hand_pose'][::stride]
        global_rot_euler = mhr['global_rot'][::stride]
        cam_t = mhr['cam_t'][::stride]
        n = body_euler.shape[0]

        joints_all = mhr['joints']
        if joints_all is not None:
            joints_all = joints_all[::stride]
            feet_idx = [13, 14, 15, 18]
            feet_joints = torch.from_numpy(joints_all[:, feet_idx, :]).float().permute(1, 0, 2)
            feetv = torch.zeros(4, n)
            feetv[:, :-1] = (feet_joints[:, 1:] - feet_joints[:, :-1]).norm(dim=-1)
            contacts = (feetv < 0.01).numpy().astype(np.float32).T

            selected = joints_all[:, SELECTED_JOINT_IDXS, :]
            jvel = np.zeros_like(selected)
            jvel[1:] = selected[1:] - selected[:-1]
            jvel = jvel.reshape(n, -1).astype(np.float32)
        else:
            contacts = np.zeros((n, 4), dtype=np.float32)
            jvel = np.zeros((n, JOINT_VEL_DIM), dtype=np.float32)

        body_cont = compact_model_params_to_cont_body(
            torch.from_numpy(body_euler).float()).numpy()
        global_rot_6d = euler3_to_rot6d(
            torch.from_numpy(global_rot_euler).float()).numpy()

        pose = np.concatenate([body_cont, hand_pca, global_rot_6d, cam_t, contacts],
                              axis=1).astype(np.float32)

        shape = mhr['shape']
        if len(shape.shape) > 1:
            shape = shape[0]
        shape_arr = np.repeat(shape.reshape(1, -1), n, axis=0)

        # audio
        audio = np.array([])
        if self.args.audio_rep is not None:
            audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace('.npz', '.wav')
            if not os.path.exists(audio_file):
                return None
            audio_raw, sr = librosa.load(audio_file)
            audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=self.args.audio_sr)
            if self.args.audio_rep == "amplitude+ctc+audio":
                audio = self._encode_audio(audio_raw)
            else:
                audio = audio_raw

        # word
        word = np.array([])
        if self.args.word_rep is not None:
            word_file = f"{self.data_dir}{self.args.word_rep}/{f_name}.TextGrid"
            if not os.path.exists(word_file):
                return None
            word = self._load_textgrid(word_file, n)

        # vid
        vid = np.repeat(np.array([-1.0]).reshape(1, 1), n, axis=0)

        # dialogue states (pre-computed at ~6.25Hz, need alignment to pose_fps)
        states = np.zeros(n, dtype=np.int64)
        if self.state_dir:
            state_file = os.path.join(self.state_dir, f_name + "_states.npy")
            if os.path.exists(state_file):
                raw_states = np.load(state_file)
                if len(raw_states) >= n:
                    states = raw_states[:n]
                elif len(raw_states) > 0:
                    indices = np.linspace(0, len(raw_states) - 1, n).astype(int)
                    states = raw_states[indices]

        return {
            'pose': pose, 'audio': audio, 'word': word, 'vid': vid,
            'trans': cam_t.copy(), 'shape': shape_arr, 'jvel': jvel,
            'states': states,
        }

    def _encode_audio(self, audio_raw):
        """Encode raw audio to amplitude+ctc+audio 3-channel representation."""
        from numpy.lib import stride_tricks
        input_values = self.processor(
            audio_raw, return_tensors="pt", sampling_rate=16000
        ).input_values
        input_values_16k = input_values.squeeze(0).numpy()

        frame_length = 1024
        shape = (audio_raw.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_raw.strides[-1], audio_raw.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_raw, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = np.pad(
            amplitude_envelope, (0, frame_length - 1),
            mode='constant', constant_values=amplitude_envelope[-1])

        ctc_array = np.zeros(len(audio_raw), dtype=float)
        audio_3ch = np.concatenate([
            amplitude_envelope.reshape(-1, 1),
            ctc_array.reshape(-1, 1),
            input_values_16k.reshape(-1, 1)
        ], axis=1)

        with torch.no_grad():
            logits = self.wav2vec_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1).squeeze().numpy()
        audio_length = len(audio_3ch)
        predicted_length = len(predicted_ids)
        indices = np.linspace(0, audio_length, num=predicted_length, endpoint=False).astype(int)
        expanded = np.zeros(audio_length, dtype=predicted_ids.dtype)
        for ii in range(predicted_length - 1):
            expanded[indices[ii]:indices[ii + 1]] = predicted_ids[ii]
        expanded[indices[-1]:] = predicted_ids[-1]
        audio_3ch[:, 1] += expanded
        return audio_3ch

    def _load_textgrid(self, word_file, n):
        """Load TextGrid word alignment and convert to token indices."""
        tgrid = tg.TextGrid.fromFile(word_file)
        words = []
        for i in range(n):
            current_time = i / self.args.pose_fps
            found = False
            for word in tgrid[0]:
                if word.minTime <= current_time <= word.maxTime:
                    if word.mark == " ":
                        words.append(self.lang_model.PAD_token)
                    else:
                        words.append(self.lang_model.get_word_index(word.mark))
                    found = True
                    break
            if not found:
                words.append(self.lang_model.UNK_token)
        return np.array(words)

    def cache_generation(self, out_lmdb_dir, disable_filtering,
                         clean_first_seconds, clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 50))
        n_filtered_out = defaultdict(int)
        skipped_indices = []

        for index, file_row in self.selected_file.iterrows():
            f_name1 = file_row["id"]
            f_name2 = f_name1.rsplit("speaker1", 1)[0] + "speaker2"

            logger.info(colored(
                f"# ---- Building dual cache for {f_name1} <-> {f_name2} ---- #", "blue"))

            data1 = self._load_speaker(f_name1)
            if data1 is None:
                logger.warning(f"Skipping: speaker1 missing for {f_name1}")
                skipped_indices.append(index)
                continue
            data2 = self._load_speaker(f_name2)
            if data2 is None:
                logger.warning(f"Skipping: speaker2 missing for {f_name2}")
                skipped_indices.append(index)
                continue

            n = min(data1['pose'].shape[0], data2['pose'].shape[0])
            for key in ('pose', 'trans', 'shape', 'vid', 'jvel', 'states'):
                data1[key] = data1[key][:n]
                data2[key] = data2[key][:n]
            if len(data1['word']) > 0:
                data1['word'] = data1['word'][:n]
            if len(data2['word']) > 0:
                data2['word'] = data2['word'][:n]

            filtered = self._sample_from_clip(
                dst_lmdb_env, data1, data2,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test)
            for t in filtered:
                n_filtered_out[t] += filtered[t]

        if skipped_indices:
            self.selected_file = self.selected_file.drop(skipped_indices).reset_index(drop=True)
            logger.info(colored(
                f"Dropped {len(skipped_indices)} skipped pairs from selected_file, "
                f"now {len(self.selected_file)} entries", "yellow"))

        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of dual samples: {txn.stat()['entries']}", "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def _sample_from_clip(self, dst_lmdb_env, data1, data2,
                          disable_filtering, clean_first_seconds,
                          clean_final_seconds, is_test):
        n_pose = min(data1['pose'].shape[0], data2['pose'].shape[0])
        round_seconds = n_pose // self.args.pose_fps

        has_audio = isinstance(data1['audio'], np.ndarray) and data1['audio'].size > 1
        if has_audio:
            rs_a1 = data1['audio'].shape[0] // self.args.audio_sr
            rs_a2 = data2['audio'].shape[0] // self.args.audio_sr
            round_seconds = min(round_seconds, rs_a1, rs_a2)

        clip_s_t = clean_first_seconds
        clip_e_t = round_seconds - clean_final_seconds
        clip_s_f_audio = self.args.audio_fps * clip_s_t
        clip_s_f_pose = clip_s_t * self.args.pose_fps
        clip_e_f_pose = clip_e_t * self.args.pose_fps

        n_filtered_out = defaultdict(int)

        for ratio in self.args.multi_length_training:
            if is_test:
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio * self.ori_stride)
                cut_length = int(self.ori_length * ratio)

            num_sub = math.floor(
                (clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            audio_short = (math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                           if has_audio else 0)

            sample_list = []
            for i in range(num_sub):
                s = clip_s_f_pose + i * self.args.stride
                e = s + cut_length

                sp1, sp2 = data1['pose'][s:e], data2['pose'][s:e]
                if len(sp1) != cut_length or len(sp2) != cut_length:
                    continue

                st1, st2 = data1['trans'][s:e], data2['trans'][s:e]
                ss1, ss2 = data1['shape'][s:e], data2['shape'][s:e]
                sj1, sj2 = data1['jvel'][s:e], data2['jvel'][s:e]
                sv1 = data1['vid'][s:e] if len(data1['vid']) > 0 else np.array([-1])
                sv2 = data2['vid'][s:e] if len(data2['vid']) > 0 else np.array([-1])
                sw1 = data1['word'][s:e] if len(data1['word']) > 0 else np.array([-1])
                sw2 = data2['word'][s:e] if len(data2['word']) > 0 else np.array([-1])
                sstate1 = data1['states'][s:e]
                sstate2 = data2['states'][s:e]

                if has_audio:
                    a_s = int(clip_s_f_audio + math.floor(
                        i * self.args.stride * self.args.audio_fps / self.args.pose_fps))
                    sa1 = data1['audio'][a_s:a_s + audio_short]
                    sa2 = data2['audio'][a_s:a_s + audio_short]
                else:
                    sa1 = np.array([-1])
                    sa2 = np.array([-1])

                # Direction A: speaker1 is condition, speaker2 is target
                sample_list.append((
                    sp1, sa1, sw1, sv1, st1, ss1, sj1, sstate1,
                    sp2, sa2, sw2, sv2, st2, ss2, sj2, sstate2,
                ))
                if not is_test:
                    # Direction B: speaker2 is condition, speaker1 is target
                    sample_list.append((
                        sp2, sa2, sw2, sv2, st2, ss2, sj2, sstate2,
                        sp1, sa1, sw1, sv1, st1, ss1, sj1, sstate1,
                    ))

            if sample_list:
                with dst_lmdb_env.begin(write=True) as txn:
                    for entry in sample_list:
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = pyarrow.serialize(list(entry)).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = pyarrow.deserialize(txn.get(key))

            (pose1, audio1, word1, vid1, trans1, shape1, jvel1, state1,
             pose2, audio2, word2, vid2, trans2, shape2, jvel2, state2) = sample

            def to_tensor(arr, dtype=torch.float32, reshape=True):
                t = torch.from_numpy(np.array(arr, copy=True))
                if reshape and t.ndim == 2:
                    t = t.reshape((t.shape[0], -1))
                return t.to(dtype)

            is_test = self.loader_type == "test"
            return {
                "pose1": to_tensor(pose1),
                "audio1": to_tensor(audio1),
                "word1": to_tensor(word1, torch.int32, reshape=False),
                "id1": to_tensor(vid1),
                "trans1": to_tensor(trans1),
                "beta1": to_tensor(shape1),
                "jvel1": to_tensor(jvel1),
                "state1": to_tensor(state1, torch.long, reshape=False),
                "pose2": to_tensor(pose2),
                "audio2": to_tensor(audio2),
                "word2": to_tensor(word2, torch.int32, reshape=False),
                "id2": to_tensor(vid2),
                "trans2": to_tensor(trans2),
                "beta2": to_tensor(shape2),
                "jvel2": to_tensor(jvel2),
                "state2": to_tensor(state2, torch.long, reshape=False),
            }
