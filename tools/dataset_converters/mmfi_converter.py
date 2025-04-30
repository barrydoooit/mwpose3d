from copy import deepcopy
import os
from typing import List, Literal
import yaml
import pickle
from pathlib import Path

import numpy as np
import h5py



DEFAULT_CONFIG = {
    "protocol": "protocol1",
    "data_unit": "frame",
    "random_split": {
        "ratio": 0.8,
        "random_seed": 0,
        "train_dataset": {
            "split": "training",
            "scenes": None,
            "subjects": None,
            "actions": "all"
        },
        "val_dataset": {
            "split": "validation",
            "scenes": None,
            "subjects": None,
            "actions": "all"
        }
    },
    "cross_scene_split": {
        "train_dataset": {
            "split": "training",
            "scenes": ["E01", "E02", "E03"],
            "subjects": None,
            "actions": "all"
        },
        "val_dataset": {
            "split": "validation",
            "scenes": ["E04"],
            "subjects": None,
            "actions": "all"
        }
    },
    "cross_subject_split": {
        "train_dataset": {
            "split": "training",
            "scenes": None,
            "subjects": ["S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09", "S11", "S12", "S13", "S14", "S16", "S17", "S18", "S19", "S21", "S22", "S23", "S24", "S26", "S27", "S28", "S29", "S31", "S32", "S33", "S34", "S36", "S37", "S38", "S39"],
            "actions": "all"
        },
        "val_dataset": {
            "split": "validation",
            "scenes": None,
            "subjects": ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"],
            "actions": "all"
        }
    },
    "manual_split": {
        "train_dataset": {
            "split": "training",
            "scenes": None,
            "subjects": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40"],
            "actions": ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21"]
        },
        "val_dataset": {
            "split": "validation",
            "scenes": None,
            "subjects": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40"],
            "actions": ["A22", "A23", "A24", "A25", "A26", "A27"]
        }
    },
    "split_to_use": "cross_subject_split"
}

class MMFiDatasetConverter:
    def __init__(self,
                 input_root: Path,
                 output_root: Path,
                 modality: List[Literal['mmwave', 'mmwave_filtered']] = ['mmwave', 'mmwave_filtered'],
                 config: dict = DEFAULT_CONFIG):
        self.input_root = input_root
        self.output_root = output_root
        self.modality = modality
        self.config = deepcopy(config)

        self.out_mmwaves = [self.output_root / prefix for prefix in self.modality]
        self.out_skeleton = self.output_root / 'skeleton'
        for out_mmwave in self.out_mmwaves:
            out_mmwave.mkdir(parents=True, exist_ok=True)
        self.out_skeleton.mkdir(parents=True, exist_ok=True)

        self.all_subjects = [f"S{i:02d}" for i in range(1, 41)]
        self.all_actions = [f"A{i:02d}" for i in range(1, 28)]
    
        self.info_prefix = 'info'
        self.train_form, self.val_form = self._decode_config()
        self.records = {'train': [], 'val': []}
    
    def _decode_config(self):
        cfg = self.config
        protocol = cfg.get('protocol', None)
        split_to_use = cfg.get('split_to_use', None)

        if protocol == 'protocol1':
            actions = ['A02','A03','A04','A05','A13','A14','A17','A18','A19','A20','A21','A22','A23','A27']
        elif protocol == 'protocol2':
            actions = ['A01','A06','A07','A08','A09','A10','A11','A12','A15','A16','A24','A25','A26']
        else:
            actions = list(self.all_actions)

        train_form = {}
        val_form = {}

        if split_to_use == 'random_split':
            self.info_prefix = 'info_rand'
            rs = cfg['random_split']['random_seed']
            ratio = cfg['random_split']['ratio']
            for action in actions:
                np.random.seed(rs)
                perm = np.random.permutation(len(self.all_subjects))
                n_train = int(np.floor(ratio * len(self.all_subjects)))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                train_subjs = [self.all_subjects[i] for i in train_idx]
                val_subjs = [self.all_subjects[i] for i in val_idx]

                for subj in self.all_subjects:
                    if subj in train_subjs:
                        train_form.setdefault(subj, []).append(action)
                    if subj in val_subjs:
                        val_form.setdefault(subj, []).append(action)
                rs += 1
        
        elif split_to_use == 'cross_scene_split':
            self.info_prefix = 'info_scene'
            train_subjs = [f"S{i:02d}" for i in range(1, 31)]
            val_subjs = [f"S{i:02d}" for i in range(31, 41)]
            for subj in train_subjs:
                train_form[subj] = list(actions)
            for subj in val_subjs:
                val_form[subj] = list(actions)
        
        elif split_to_use == 'cross_subject_split':
            self.info_prefix = 'info_subj'
            ts = cfg['cross_subject_split']['train_dataset']['subjects']
            vs = cfg['cross_subject_split']['val_dataset']['subjects']
            for subj in ts:
                train_form[subj] = list(actions)
            for subj in vs:
                val_form[subj] = list(actions)

        else: # manual_split
            self.info_prefix = 'info_manual'
            ts = cfg['manual_split']['train_dataset']['subjects']
            ta = cfg['manual_split']['train_dataset']['actions']
            vs = cfg['manual_split']['val_dataset']['subjects']
            va = cfg['manual_split']['val_dataset']['actions']
            for subj in ts:
                train_form[subj] = list(ta)
            for subj in vs:
                val_form[subj] = list(va)
        
        return train_form, val_form

    @staticmethod
    def _get_scene(subject: str) -> str:
        """Map S01-S40 to E01-E04 exactly as in MMFi."""
        idx = int(subject[1:])
        if 1 <= idx <= 10:
            return 'E01'
        elif 11 <= idx <= 20:
            return 'E02'
        elif 21 <= idx <= 30:
            return 'E03'
        elif 31 <= idx <= 40:
            return 'E04'
        else:
            raise ValueError(f"Unknown subject: {subject}")
    
    EXPECTED_FRAMES = 297
    @staticmethod
    def load_mmwave(mmwave_dir: Path):
        bin_files = sorted(mmwave_dir.glob('frame*.bin'))
        pts_list = []
        idx = [0]
        frame_idx = []
        for bf in bin_files:
            num = int(bf.stem.replace('frame',''))
            raw = bf.read_bytes()
            pts = np.frombuffer(raw, dtype=np.float64).copy().reshape(-1,5)
            pts_list.append(pts)
            idx.append(idx[-1] + pts.shape[0])
            frame_idx.append(num)
        pcd_data = np.concatenate(pts_list,axis=0) if pts_list else np.zeros((0,5),dtype=np.float64)
        pcd_idx  = np.array(idx,dtype=np.int64)
        # Swap snr and vel columns. NOTE: MMFi stated that the order is x, y, z, vel, snr, but the data seems to be reversed in vel and snr.
        if pcd_data.size > 0:
            pcd_data[:, [3, 4]] = pcd_data[:, [4, 3]]
        pcd_cols = ['x', 'y', 'z', 'vel', 'snr']

        # check if frames are contiguous from min to max
        if frame_idx and frame_idx == list(range(frame_idx[0], frame_idx[0]+len(frame_idx))):
            frame_idx_arr = None
        else:
            frame_idx_arr = np.array(frame_idx, dtype=np.int16)
        return pcd_data, pcd_idx, pcd_cols, frame_idx_arr
    
    def write_sequence(self,
                       scene: str,
                       subject: str,
                       action: str,
                       split: str):
        seq_dir = self.input_root / scene / subject / action
        skel = np.load(seq_dir / 'ground_truth.npy')
        fname = f"{scene}_{subject}_{action}.h5"

        record = {
                'scene': scene,
                'subject': subject,
                'action': action,
        }
        frame_counts = {}

        for modality in self.modality:
            pcd_data, pcd_idx, pcd_cols, frame_idx = self.load_mmwave(seq_dir / modality)
            pm = self.output_root / modality / fname
            with h5py.File(pm, 'w') as h5f:
                grp = h5f.create_group('pcd')
                ds = grp.create_dataset('data', data=pcd_data)
                ds.attrs['columns'] = pcd_cols
                grp.create_dataset('index', data=pcd_idx)
                if frame_idx is not None:
                    grp.create_dataset('frame_idx', data=frame_idx)
            record[f"{modality}_path"] = pm.name
            frame_counts[modality] = int(pcd_idx.shape[0] - 1)

        ps = self.out_skeleton / fname
        skel_flattened = self.joint_remap(skel, flatten=True)
        assert skel_flattened.shape[1] == 17 * 3, f"Skeleton data should have 17 joints, but got {skel_flattened.shape[0] // 3} joints."
        with h5py.File(ps, 'w') as h5f:
            h5f.create_dataset('skel', data=skel_flattened)
        record[f"skeleton_path"] = ps.name
        record['frame_count'] = frame_counts
        

        self.records[split].append(record)
    
    MMFI_KEYPOINT_TYPE = {
        "SPINE_BASE": 0,
        "HIP_RIGHT": 1,
        "KNEE_RIGHT": 2,
        "ANKLE_RIGHT": 3,
        "HIP_LEFT": 4,
        "KNEE_LEFT": 5,
        "ANKLE_LEFT": 6,
        "SPINE_MID": 7,
        "SPINE_SHOULDER": 8,
        "NECK": 9,
        "HEAD": 10,
        "SHOULDER_LEFT": 11,
        "ELBOW_LEFT": 12,
        "WRIST_LEFT": 13,
        "SHOULDER_RIGHT": 14,
        "ELBOW_RIGHT": 15,
        "WRIST_RIGHT": 16,
    }
    
    def joint_remap(self, skel_array_2d: np.ndarray, flatten: bool = True):
        skel_array = skel_array_2d.copy()
        skel_array[:, :, 0] = -skel_array[:, :, 0].copy()
        skel_array[:, :, 1] = skel_array[:, :, 2].copy()
        skel_array[:, :, 2] = -skel_array[:, :, 1].copy()
        if flatten:
            skel_array = skel_array.reshape(skel_array.shape[0], -1)
        return skel_array
    
    def process_split(self, split: str, form: dict):
        for subject, actions in form.items():
            scene = self._get_scene(subject)
            for action in actions:
                self.write_sequence(scene, subject, action, split)
        
        with open(self.output_root / f'{self.info_prefix}_{split}.pkl', 'wb') as f:
            pickle.dump(self.records[split], f)
    
    def process_all(self):
        self.process_split('train', self.train_form)
        self.process_split('val', self.val_form)

