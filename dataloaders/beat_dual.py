import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyarrow
import librosa
from numpy.lib import stride_tricks

from .build_vocab import Vocab
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools
from .beat_sep_lower import load_mhr_native, MotionPreprocessor


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type

        self.rank = dist.get_rank()
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0, 0]

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        self.joints = 55
        self.mhr_body_dim = 130
        self.mhr_hand_dim = 108
        self.mhr_face_dim = 75
        self.mhr_global_dim = 7
        self.mhr_total_dim = self.mhr_body_dim + self.mhr_hand_dim + self.mhr_face_dim + self.mhr_global_dim

        split_rule = pd.read_csv(args.data_path + "train_test_split.csv")
        selected = split_rule.loc[split_rule['type'] == loader_type]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[split_rule['type'] == 'additional']
            selected = pd.concat([selected, split_b])
        # Keep only speaker1 rows to avoid duplicate pairs
        self.selected_file = selected[selected['id'].str.endswith('speaker1')]
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty after speaker1 filtering, use train set 0-8 instead")
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

        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_dual_cache"

        if self.args.beat_align:
            if not os.path.exists(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy")

        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

    def calculate_mean_velocity(self, save_path):
        dir_p = self.data_dir + self.args.pose_rep + "/"
        all_list = []
        from tqdm import tqdm
        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                mhr = load_mhr_native(dir_p + tar)
                joints_np = mhr['joints']
                if joints_np is None:
                    continue
                joints_np = joints_np[:, :55, :]
                n = joints_np.shape[0]
                joints = torch.from_numpy(joints_np).float().reshape(n, 55 * 3).permute(1, 0)
                dt = 1 / 30
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                vel_joints_np = np.linalg.norm(vel_seq.numpy(), axis=2)
                all_list.append(vel_joints_np)
        avg_vel = np.mean(np.concatenate(all_list, axis=0), axis=0)
        np.save(save_path, avg_vel)

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

    def idmapping(self, id):
        if id == 30: id = 8
        if id == 28: id = 14
        if id == 27: id = 19
        return id - 1

    def _process_single_speaker(self, f_name):
        """Load and process MHR pose, audio, and word data for one speaker.

        Returns a dict with keys: pose, trans, shape, audio, word, vid, facial,
        or None if a required file is missing.
        """
        ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
        pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
        id_pose = f_name

        if not os.path.exists(pose_file):
            logger.warning(f"Pose file not found: {pose_file}")
            return None

        # ---- Pose ----
        if "smplx" in self.args.pose_rep:
            mhr = load_mhr_native(pose_file)
            assert 30 % self.args.pose_fps == 0
            stride = int(30 / self.args.pose_fps)

            body_pose = mhr['body_pose'][::stride]
            jaw = mhr['jaw'][::stride]
            hand_pose = mhr['hand_pose'][::stride]
            expr = mhr['expr'][::stride]
            global_rot = mhr['global_rot'][::stride]
            cam_t = mhr['cam_t'][::stride]
            n = body_pose.shape[0]

            joints_all = mhr['joints']
            if joints_all is not None:
                joints_all = joints_all[::stride]
                feet_idx = [13, 14, 15, 18]
                feet_joints = torch.from_numpy(joints_all[:, feet_idx, :]).float().permute(1, 0, 2)
                feetv = torch.zeros(4, n)
                feetv[:, :-1] = (feet_joints[:, 1:] - feet_joints[:, :-1]).norm(dim=-1)
                contacts = (feetv < 0.01).numpy().astype(float).T
            else:
                contacts = np.zeros((n, 4), dtype=np.float32)

            face_params = np.concatenate([expr, jaw], axis=1)
            global_params = np.concatenate([global_rot, contacts], axis=1)
            pose = np.concatenate([body_pose, hand_pose, face_params, global_params], axis=1).astype(np.float32)
            trans = cam_t.copy()
            shape = mhr['shape']
            if len(shape.shape) > 1:
                shape = shape[0]
            shape = np.repeat(shape.reshape(1, -1), n, axis=0)
            facial = face_params if self.args.facial_rep is not None else np.array([])
        else:
            logger.error("beat_dual only supports smplx pose_rep")
            return None

        # ---- Speaker ID ----
        try:
            int_value = self.idmapping(int(f_name.split("_")[0]))
        except (ValueError, IndexError):
            int_value = 0
        if self.args.id_rep is not None:
            vid = np.repeat(np.array(int_value).reshape(1, 1), pose.shape[0], axis=0)
        else:
            vid = np.array([])

        # ---- Audio ----
        audio = np.array([])
        if self.args.audio_rep is not None:
            logger.info(f"# ---- Building cache for Audio  {id_pose} ---- #")
            audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
            if not os.path.exists(audio_file):
                logger.warning(f"Audio file not found: {audio_file}")
                return None
            audio_raw, sr = librosa.load(audio_file)
            audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=self.args.audio_sr)

            if self.args.audio_rep == "amplitude+ctc+audio":
                input_values = self.processor(audio_raw, return_tensors="pt", sampling_rate=16000).input_values
                input_values_16k = input_values.squeeze(0).numpy()

                frame_length = 1024
                shape_a = (audio_raw.shape[-1] - frame_length + 1, frame_length)
                strides_a = (audio_raw.strides[-1], audio_raw.strides[-1])
                rolling_view = stride_tricks.as_strided(audio_raw, shape=shape_a, strides=strides_a)
                amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length - 1),
                                            mode='constant', constant_values=amplitude_envelope[-1])
                ctc_array = np.zeros(len(audio_raw), dtype=float)
                audio = np.concatenate([
                    amplitude_envelope.reshape(-1, 1),
                    ctc_array.reshape(-1, 1),
                    input_values_16k.reshape(-1, 1)
                ], axis=1)

                with torch.no_grad():
                    logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1).squeeze().numpy()
                audio_length = len(audio)
                predicted_length = len(predicted_ids)
                indices = np.linspace(0, audio_length, num=predicted_length, endpoint=False).astype(int)
                expanded_predicted_ids = np.zeros(audio_length, dtype=predicted_ids.dtype)
                for i in range(predicted_length - 1):
                    expanded_predicted_ids[indices[i]:indices[i + 1]] = predicted_ids[i]
                expanded_predicted_ids[indices[-1]:] = predicted_ids[-1]
                audio[:, 1] += expanded_predicted_ids
            elif self.args.audio_rep == "onset+amplitude":
                frame_length = 1024
                shape_a = (audio_raw.shape[-1] - frame_length + 1, frame_length)
                strides_a = (audio_raw.strides[-1], audio_raw.strides[-1])
                rolling_view = stride_tricks.as_strided(audio_raw, shape=shape_a, strides=strides_a)
                amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length - 1),
                                            mode='constant', constant_values=amplitude_envelope[-1])
                audio_onset_f = librosa.onset.onset_detect(y=audio_raw, sr=self.args.audio_sr, units='frames')
                onset_array = np.zeros(len(audio_raw), dtype=float)
                onset_array[audio_onset_f] = 1.0
                audio = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
            elif self.args.audio_rep == "mfcc":
                audio = librosa.feature.melspectrogram(
                    y=audio_raw, sr=self.args.audio_sr, n_mels=128,
                    hop_length=int(self.args.audio_sr / self.args.audio_fps))
                audio = audio.transpose(1, 0)
            else:
                audio = audio_raw

        # ---- Word ----
        word = np.array([])
        time_offset = 0
        if self.args.word_rep is not None:
            logger.info(f"# ---- Building cache for Word   {id_pose} ---- #")
            word_file = f"{self.data_dir}{self.args.word_rep}/{id_pose}.TextGrid"
            if not os.path.exists(word_file):
                logger.warning(f"Word file not found: {word_file}")
                return None
            tgrid = tg.TextGrid.fromFile(word_file)
            word_list = []
            for i in range(pose.shape[0]):
                found_flag = False
                current_time = i / self.args.pose_fps + time_offset
                for j, w in enumerate(tgrid[0]):
                    if w.minTime <= current_time <= w.maxTime:
                        if w.mark == " ":
                            word_list.append(self.lang_model.PAD_token)
                        else:
                            word_list.append(self.lang_model.get_word_index(w.mark))
                        found_flag = True
                        break
                if not found_flag:
                    word_list.append(self.lang_model.UNK_token)
            word = np.array(word_list)

        return {
            'pose': pose, 'trans': trans, 'shape': shape,
            'audio': audio, 'word': word, 'vid': vid, 'facial': facial,
        }

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds, clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        if len(self.args.training_speakers) == 1:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 50))
        else:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 200))
        n_filtered_out = defaultdict(int)

        for index, file_name in self.selected_file.iterrows():
            f_name1 = file_name["id"]
            f_name2 = f_name1.rsplit("speaker1", 1)[0] + "speaker2"

            logger.info(colored(f"# ---- Building dual cache for {f_name1} <-> {f_name2} ---- #", "blue"))

            data1 = self._process_single_speaker(f_name1)
            if data1 is None:
                logger.warning(f"Skipping pair: speaker1 data missing for {f_name1}")
                continue
            data2 = self._process_single_speaker(f_name2)
            if data2 is None:
                logger.warning(f"Skipping pair: speaker2 data missing for {f_name2}")
                continue

            # Align frame count across both speakers
            n = min(data1['pose'].shape[0], data2['pose'].shape[0])
            data1['pose'] = data1['pose'][:n]
            data1['trans'] = data1['trans'][:n]
            data1['shape'] = data1['shape'][:n]
            data2['pose'] = data2['pose'][:n]
            data2['trans'] = data2['trans'][:n]
            data2['shape'] = data2['shape'][:n]
            if len(data1['vid']) > 0:
                data1['vid'] = data1['vid'][:n]
            if len(data2['vid']) > 0:
                data2['vid'] = data2['vid'][:n]
            if len(data1['word']) > 0:
                data1['word'] = data1['word'][:n]
            if len(data2['word']) > 0:
                data2['word'] = data2['word'][:n]
            if len(data1['facial']) > 0:
                data1['facial'] = data1['facial'][:n]
            if len(data2['facial']) > 0:
                data2['facial'] = data2['facial'][:n]

            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                data1['pose'], data1['audio'], data1['word'], data1['vid'],
                data1['trans'], data1['shape'],
                data2['pose'], data2['audio'], data2['word'], data2['vid'],
                data2['trans'], data2['shape'], data2['facial'],
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
            )
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]

        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            total = txn.stat()["entries"] + n_total_filtered
            if total > 0:
                logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                    n_total_filtered, 100 * n_total_filtered / total), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def _sample_from_clip(
        self, dst_lmdb_env,
        pose1, audio1, word1, vid1, trans1, shape1,
        pose2, audio2, word2, vid2, trans2, shape2, facial2,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
    ):
        round_seconds_skeleton = min(
            pose1.shape[0] // self.args.pose_fps,
            pose2.shape[0] // self.args.pose_fps,
        )

        if len(audio1) > 0 and len(audio2) > 0:
            if self.args.audio_rep not in ("wave16k", "mfcc"):
                rs_audio1 = len(audio1) // self.args.audio_fps
                rs_audio2 = len(audio2) // self.args.audio_fps
            elif self.args.audio_rep == "mfcc":
                rs_audio1 = audio1.shape[0] // self.args.audio_fps
                rs_audio2 = audio2.shape[0] // self.args.audio_fps
            else:
                rs_audio1 = audio1.shape[0] // self.args.audio_sr
                rs_audio2 = audio2.shape[0] // self.args.audio_sr
            round_seconds_audio = min(rs_audio1, rs_audio2)
            logger.info(f"pose: {round_seconds_skeleton}s, audio1: {rs_audio1}s, audio2: {rs_audio2}s")
            round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)

        clip_s_t = clean_first_seconds
        clip_e_t = round_seconds_skeleton - clean_final_seconds
        clip_s_f_audio = self.args.audio_fps * clip_s_t
        clip_e_f_audio = clip_e_t * self.args.audio_fps
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
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips expected with stride {self.args.stride}")

            audio_short_length = 0
            if len(audio1) > 0:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")

            sample_list = []

            for i in range(num_subdivision):
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length

                sp1 = pose1[start_idx:fin_idx]
                st1 = trans1[start_idx:fin_idx]
                ss1 = shape1[start_idx:fin_idx]
                sp2 = pose2[start_idx:fin_idx]
                st2 = trans2[start_idx:fin_idx]
                ss2 = shape2[start_idx:fin_idx]

                if self.args.audio_rep is not None:
                    a_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    a_end = a_start + audio_short_length
                    sa1 = audio1[a_start:a_end]
                    sa2 = audio2[a_start:a_end]
                else:
                    sa1 = np.array([-1])
                    sa2 = np.array([-1])

                sw1 = word1[start_idx:fin_idx] if len(word1) > 0 else np.array([-1])
                sw2 = word2[start_idx:fin_idx] if len(word2) > 0 else np.array([-1])
                sv1 = vid1[start_idx:fin_idx] if len(vid1) > 0 else np.array([-1])
                sv2 = vid2[start_idx:fin_idx] if len(vid2) > 0 else np.array([-1])
                sf2 = facial2[start_idx:fin_idx] if len(facial2) > 0 else np.array([-1])

                if sp1.any() is not None and sp2.any() is not None:
                    sp1, msg1 = MotionPreprocessor(sp1).get()
                    sp2, msg2 = MotionPreprocessor(sp2).get()
                    ok1 = len(sp1) > 0 or disable_filtering
                    ok2 = len(sp2) > 0 or disable_filtering
                    if ok1 and ok2:
                        sample_list.append((sp1, sa1, sw1, sv1, st1, ss1,
                                            sp2, sa2, sw2, sv2, st2, ss2, sf2))
                    else:
                        if not ok1:
                            n_filtered_out[msg1] += 1
                        if not ok2:
                            n_filtered_out[msg2] += 1

            if len(sample_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for entry in sample_list:
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = list(entry)
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            (tar_pose1, in_audio1, in_word1, vid1, trans1, in_shape1,
             tar_pose2, in_audio2, in_word2, vid2, trans2, in_shape2, in_facial2) = sample

            tar_pose1 = np.array(tar_pose1, copy=True)
            in_audio1 = np.array(in_audio1, copy=True)
            in_word1 = np.array(in_word1, copy=True)
            vid1 = np.array(vid1, copy=True)
            trans1 = np.array(trans1, copy=True)
            in_shape1 = np.array(in_shape1, copy=True)

            tar_pose2 = np.array(tar_pose2, copy=True)
            in_audio2 = np.array(in_audio2, copy=True)
            in_word2 = np.array(in_word2, copy=True)
            vid2 = np.array(vid2, copy=True)
            trans2 = np.array(trans2, copy=True)
            in_shape2 = np.array(in_shape2, copy=True)
            in_facial2 = np.array(in_facial2, copy=True)

            in_audio1 = torch.from_numpy(in_audio1).float()
            in_audio2 = torch.from_numpy(in_audio2).float()
            in_word1 = torch.from_numpy(in_word1).int()
            in_word2 = torch.from_numpy(in_word2).int()

            if self.loader_type == "test":
                tar_pose1 = torch.from_numpy(tar_pose1).float()
                trans1 = torch.from_numpy(trans1).float()
                vid1 = torch.from_numpy(vid1).float()
                in_shape1 = torch.from_numpy(in_shape1).float()
                tar_pose2 = torch.from_numpy(tar_pose2).float()
                trans2 = torch.from_numpy(trans2).float()
                vid2 = torch.from_numpy(vid2).float()
                in_shape2 = torch.from_numpy(in_shape2).float()
            else:
                tar_pose1 = torch.from_numpy(tar_pose1).reshape((tar_pose1.shape[0], -1)).float()
                trans1 = torch.from_numpy(trans1).reshape((trans1.shape[0], -1)).float()
                vid1 = torch.from_numpy(vid1).reshape((vid1.shape[0], -1)).float()
                in_shape1 = torch.from_numpy(in_shape1).reshape((in_shape1.shape[0], -1)).float()
                tar_pose2 = torch.from_numpy(tar_pose2).reshape((tar_pose2.shape[0], -1)).float()
                trans2 = torch.from_numpy(trans2).reshape((trans2.shape[0], -1)).float()
                vid2 = torch.from_numpy(vid2).reshape((vid2.shape[0], -1)).float()
                in_shape2 = torch.from_numpy(in_shape2).reshape((in_shape2.shape[0], -1)).float()

            return {
                "pose1": tar_pose1, "audio1": in_audio1, "word1": in_word1,
                "id1": vid1, "trans1": trans1, "beta1": in_shape1,
                "pose2": tar_pose2, "audio2": in_audio2, "word2": in_word2,
                "id2": vid2, "trans2": trans2, "beta2": in_shape2,
            }
