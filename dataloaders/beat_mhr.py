"""
MHR DataLoader for MambaTalk_origin.
Loads MHR npz data, converts body Euler → 6D continuous representation.
Cache layout: body_cont(260) + hand(108) + global_rot_6d(6) + cam_t(3) + contact(4) = 381d
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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

BODY_CONT_DIM = 260
HAND_DIM = 108
GLOBAL_ROT6D_DIM = 6
CAM_T_DIM = 3
CONTACT_DIM = 4
TOTAL_DIM = BODY_CONT_DIM + HAND_DIM + GLOBAL_ROT6D_DIM + CAM_T_DIM + CONTACT_DIM  # 381

# Upper-body + hand joints selected from MHR 127 skeleton for periodicity disentanglement.
# Mapping verified via MHR70 keypoint spatial matching (dist=0.0).
SELECTED_JOINT_IDXS = [
    # Body upper (9): spine1, spine2, spine3/chest, collar, r_shoulder, r_lowarm, l_shoulder, l_lowarm, neck
    35, 36, 37, 38, 39, 40, 75, 76, 110,
    # Right hand (21): wrist + 5 fingers × 4 bones
    41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64,
    # Left hand (21): wrist + 5 fingers × 4 bones
    77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100,
]
JOINT_VEL_DIM = len(SELECTED_JOINT_IDXS) * 3  # 51 * 3 = 153


def load_mhr_native(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data['fps'][0]) if 'fps' in data.files else 30.0
    return {
        'body_pose': data['body_pose_params'].astype(np.float32),        # (N, 133)
        'hand_pose': data['hand_pose_params'].astype(np.float32),        # (N, 108)
        'global_rot': data['global_rot'].astype(np.float32),             # (N, 3)
        'cam_t': data['pred_cam_t'].astype(np.float32),                  # (N, 3)
        'shape': data['shape_params'].astype(np.float32),                # (N, 45) or (45,)
        'joints': data['pred_joint_coords'].astype(np.float32) if 'pred_joint_coords' in data.files else None,
        'fps': fps,
    }


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
        self.selected_file = split_rule.loc[split_rule['type'] == loader_type]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[split_rule['type'] == 'additional']
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty, using train set 0-8")
            self.selected_file = split_rule.loc[split_rule['type'] == 'train'].iloc[0:8]

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

        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"

        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

        if self.loader_type == "test" and len(self.selected_file) != self.n_samples:
            self._align_selected_file_with_cache()


    def _align_selected_file_with_cache(self):
        """Drop entries from selected_file whose required files are missing,
        so that selected_file indices align 1:1 with LMDB cache keys."""
        drop_idx = []
        for index, row in self.selected_file.iterrows():
            name = row['id']
            npz_ok = os.path.exists(self.data_dir + self.args.pose_rep + "/" + name + ".npz")
            wav_ok = self.args.audio_rep is None or os.path.exists(
                self.data_dir + "wave16k/" + name + ".wav")
            tg_ok = self.args.word_rep is None or os.path.exists(
                self.data_dir + self.args.word_rep + "/" + name + ".TextGrid")
            if not (npz_ok and wav_ok and tg_ok):
                drop_idx.append(index)
        if drop_idx:
            self.selected_file = self.selected_file.drop(drop_idx).reset_index(drop=True)
            logger.warning(
                f"Aligned selected_file with cache: dropped {len(drop_idx)} missing entries, "
                f"now {len(self.selected_file)} (cache has {self.n_samples})")

    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(preloaded_dir, True, 0, 0, is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering,
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)

    def __len__(self):
        return self.n_samples

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds, clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 50))
        n_filtered_out = defaultdict(int)
        skipped_indices = []

        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ".npz"
            if not os.path.exists(pose_file):
                logger.warning(f"Pose file not found: {pose_file}, skipping")
                skipped_indices.append(index)
                continue

            id_pose = f_name
            logger.info(colored(f"# ---- Building cache for {id_pose} ---- #", "blue"))

            # ========== Load MHR npz ==========
            mhr = load_mhr_native(pose_file)
            assert 30 % self.args.pose_fps == 0
            stride = int(30 / self.args.pose_fps)

            body_euler_full = mhr['body_pose']   # (N, 133)
            hand_pca = mhr['hand_pose']           # (N, 108)
            global_rot_euler = mhr['global_rot']  # (N, 3)
            cam_t = mhr['cam_t']                  # (N, 3)
            n_raw = body_euler_full.shape[0]

            # Subsample by stride
            body_euler_full = body_euler_full[::stride]
            hand_pca = hand_pca[::stride]
            global_rot_euler = global_rot_euler[::stride]
            cam_t = cam_t[::stride]
            n = body_euler_full.shape[0]

            # Compute foot contact and joint velocity from joint coordinates
            joints_all = mhr['joints']
            if joints_all is not None:
                joints_all = joints_all[::stride]
                feet_idx = [13, 14, 15, 18]
                feet_joints = torch.from_numpy(joints_all[:, feet_idx, :]).float().permute(1, 0, 2)
                feetv = torch.zeros(4, n)
                feetv[:, :-1] = (feet_joints[:, 1:] - feet_joints[:, :-1]).norm(dim=-1)
                contacts = (feetv < 0.01).numpy().astype(np.float32).T  # (N, 4)

                selected = joints_all[:, SELECTED_JOINT_IDXS, :]  # (N, 51, 3)
                jvel = np.zeros_like(selected)
                jvel[1:] = selected[1:] - selected[:-1]
                joint_velocity_each_file = jvel.reshape(n, -1).astype(np.float32)  # (N, 153)
            else:
                contacts = np.zeros((n, 4), dtype=np.float32)
                joint_velocity_each_file = np.zeros((n, JOINT_VEL_DIM), dtype=np.float32)

            # Convert body Euler(133d) → continuous(260d)
            body_cont = compact_model_params_to_cont_body(
                torch.from_numpy(body_euler_full).float()
            ).numpy()  # (N, 260)

            # Convert global_rot Euler(3d) → rot6d(6d)
            global_rot_6d = euler3_to_rot6d(
                torch.from_numpy(global_rot_euler).float()
            ).numpy()  # (N, 6)

            # Concatenate: body_cont(260) + hand(108) + global_rot_6d(6) + cam_t(3) + contact(4) = 381
            pose_each_file = np.concatenate([body_cont, hand_pca, global_rot_6d, cam_t, contacts], axis=1).astype(np.float32)

            trans_each_file = cam_t.copy()
            shape = mhr['shape']
            if len(shape.shape) > 1:
                shape = shape[0]
            shape_each_file = np.repeat(shape.reshape(1, -1), n, axis=0)

            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = np.repeat(np.array([-1.0]).reshape(1, 1), n, axis=0)

            # ========== Audio ==========
            audio_each_file = []
            if self.args.audio_rep is not None:
                audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace('.npz', '.wav')
                if not os.path.exists(audio_file):
                    logger.warning(f"Audio not found: {audio_file}, skipping")
                    skipped_indices.append(index)
                    continue
                audio_each_file, sr = librosa.load(audio_file)
                audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)
                if self.args.audio_rep == "amplitude+ctc+audio":
                    from numpy.lib import stride_tricks
                    input_values = self.processor(audio_each_file, return_tensors="pt", sampling_rate=16000).input_values
                    input_values_16k = input_values.squeeze(0).numpy()
                    frame_length = 1024
                    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length - 1), mode='constant', constant_values=amplitude_envelope[-1])
                    ctc_array = np.zeros(len(audio_each_file), dtype=float)
                    audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), ctc_array.reshape(-1, 1), input_values_16k.reshape(-1, 1)], axis=1)
                    with torch.no_grad():
                        logits = self.wav2vec_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1).squeeze().numpy()
                    audio_length = len(audio_each_file)
                    predicted_length = len(predicted_ids)
                    indices = np.linspace(0, audio_length, num=predicted_length, endpoint=False).astype(int)
                    expanded_predicted_ids = np.zeros(audio_length, dtype=predicted_ids.dtype)
                    for ii in range(predicted_length - 1):
                        expanded_predicted_ids[indices[ii]:indices[ii + 1]] = predicted_ids[ii]
                    expanded_predicted_ids[indices[-1]:] = predicted_ids[-1]
                    audio_each_file[:, 1] += expanded_predicted_ids

            # ========== Word (TextGrid) ==========
            time_offset = 0
            if self.args.word_rep is not None:
                word_file = f"{self.data_dir}{self.args.word_rep}/{id_pose}.TextGrid"
                if not os.path.exists(word_file):
                    logger.warning(f"TextGrid not found: {word_file}, skipping")
                    skipped_indices.append(index)
                    continue
                tgrid = tg.TextGrid.fromFile(word_file)
                for i in range(n):
                    found_flag = False
                    current_time = i / self.args.pose_fps + time_offset
                    for j, word in enumerate(tgrid[0]):
                        word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                        if word_s <= current_time <= word_e:
                            if word_n == " ":
                                word_each_file.append(self.lang_model.PAD_token)
                            else:
                                word_each_file.append(self.lang_model.get_word_index(word_n))
                            found_flag = True
                            break
                    if not found_flag:
                        word_each_file.append(self.lang_model.UNK_token)
                word_each_file = np.array(word_each_file)

            # ========== Sample clips ==========
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, trans_each_file, shape_each_file,
                facial_each_file, word_each_file, vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                joint_velocity_each_file=joint_velocity_each_file,
            )
            for t in filtered_result.keys():
                n_filtered_out[t] += filtered_result[t]

        if skipped_indices:
            self.selected_file = self.selected_file.drop(skipped_indices).reset_index(drop=True)
            logger.info(colored(
                f"Dropped {len(skipped_indices)} skipped files from selected_file, "
                f"now {len(self.selected_file)} entries", "yellow"))

        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for t, nf in n_filtered_out.items():
                logger.info("{}: {}".format(t, nf))
                n_total_filtered += nf
            total = txn.stat()["entries"] + n_total_filtered
            if total > 0:
                logger.info(colored("no. of excluded: {} ({:.1f}%)".format(
                    n_total_filtered, 100 * n_total_filtered / total), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, trans_each_file,
        shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        joint_velocity_each_file=None,
    ):
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps
        has_audio = isinstance(audio_each_file, np.ndarray) and audio_each_file.size > 1
        if has_audio:
            round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
            round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)

        clip_s_t = clean_first_seconds
        clip_e_t = round_seconds_skeleton - clean_final_seconds
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

            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose frames [{clip_s_f_pose}, {clip_e_f_pose}), length {cut_length}, {num_subdivision} clips")

            has_audio = isinstance(audio_each_file, np.ndarray) and audio_each_file.size > 1
            if has_audio:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)

            sample_lists = {k: [] for k in ['pose', 'audio', 'facial', 'shape', 'word', 'vid', 'emo', 'sem', 'trans', 'jvel']}

            for i in range(num_subdivision):
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length
                sample_pose = pose_each_file[start_idx:fin_idx]
                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]

                if self.args.audio_rep is not None:
                    audio_start = int(clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps))
                    sample_audio = audio_each_file[audio_start:audio_start + audio_short_length]
                else:
                    sample_audio = np.array([-1])

                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                sample_facial = np.array([-1])
                sample_emo = np.array([-1])
                sample_sem = np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if vid_each_file is not None and len(vid_each_file) > 0 else np.array([-1])

                sample_jvel = joint_velocity_each_file[start_idx:fin_idx] if joint_velocity_each_file is not None else np.zeros((cut_length, JOINT_VEL_DIM), dtype=np.float32)

                if sample_pose is not None and len(sample_pose) == cut_length:
                    sample_lists['pose'].append(sample_pose)
                    sample_lists['audio'].append(sample_audio)
                    sample_lists['facial'].append(sample_facial)
                    sample_lists['shape'].append(sample_shape)
                    sample_lists['word'].append(sample_word)
                    sample_lists['vid'].append(sample_vid)
                    sample_lists['emo'].append(sample_emo)
                    sample_lists['sem'].append(sample_sem)
                    sample_lists['trans'].append(sample_trans)
                    sample_lists['jvel'].append(sample_jvel)

            if len(sample_lists['pose']) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio, facial, shape, word, vid, emo, sem, trans, jvel in zip(
                        sample_lists['pose'], sample_lists['audio'], sample_lists['facial'],
                        sample_lists['shape'], sample_lists['word'], sample_lists['vid'],
                        sample_lists['emo'], sample_lists['sem'], sample_lists['trans'],
                        sample_lists['jvel'],
                    ):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose, audio, facial, shape, word, emo, sem, vid, trans, jvel]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)

            if len(sample) >= 10:
                tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans, jvel = sample
            else:
                tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
                jvel = np.zeros((tar_pose.shape[0], JOINT_VEL_DIM), dtype=np.float32)

            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float()
            in_audio = torch.from_numpy(in_audio).float()
            in_word = torch.from_numpy(in_word).int()
            joint_velocity = torch.from_numpy(jvel).float()

            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).float()

            return {
                "pose": tar_pose, "audio": in_audio, "facial": in_facial,
                "beta": in_shape, "word": in_word, "id": vid,
                "emo": emo, "sem": sem, "trans": trans,
                "joint_velocity": joint_velocity,
            }
