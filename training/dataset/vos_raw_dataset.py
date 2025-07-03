import random
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
    OCIDSegmentLoader,
    JacquardSegmentLoader,
    GraspNetSegmentLoader,
    RealworldSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            # subset = os.listdir(self.img_folder)
            subset = [folder for folder in os.listdir(self.img_folder) if os.path.isdir(os.path.join(self.img_folder, folder))]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        # print('1')

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg"))) #xxx.jpg list
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)



class OCIDDataset(VOSRawDataset):
    def __init__(
        self,
        dataset_folder,
        num_frames=1,
    ):
        self.dataset_folder = dataset_folder
        self.num_frames = num_frames

        training_txt = os.path.join(dataset_folder)
        image_paths = []
        with open(training_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 以逗号分隔 subfolder 和文件名
                parts = line.split(",")
                if len(parts) != 2:
                    print(f"警告：跳过格式不正确的行：{line}")
                    continue
                subfolder, filename = parts[0], parts[1]

                # 如果你需要对序列号做转换，例如将 "seq01" 替换成 "seq06"
                # 你可以在这里做处理，例如：
                # subfolder = subfolder.replace("seq01", "seq06")

                full_path = os.path.join(os.path.dirname(os.path.dirname(dataset_folder)), subfolder, "rgb", filename)
                image_paths.append(full_path)
        self.image_paths = image_paths

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_frame_path = self.image_paths[idx]

        video_label_path = video_frame_path.replace('rgb', 'Annotations').replace('png','txt')

        segment_loader = OCIDSegmentLoader(
            video_label_path=video_label_path,
            video_frame_path=video_frame_path,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_frame_path, int(idx), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.image_paths)































class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

class JacquardDataset(VOSRawDataset):
    def __init__(
        self,
        dataset_root_path,  # Path to the root of Jacquard dataset
        mode: str = 'train', # 'train' or 'test'
        split_save_dir: Optional[str] = None, # Directory to save/load split files
        split_ratio: float = 0.8, # Ratio for training set
        random_seed: int = 42, # Seed for reproducible splits
        num_frames=1, # For Jacquard, each sample is effectively a single "frame" or view
    ):
        super().__init__()
        self.dataset_root_path = dataset_root_path
        self.mode = mode.lower()
        if self.mode not in ['train', 'test']:
            raise ValueError(f"Mode must be 'train' or 'test', got {mode}")
        
        if split_save_dir is None:
            self.split_save_dir = os.path.join(self.dataset_root_path, ".dataset_splits")
        else:
            self.split_save_dir = split_save_dir
        
        os.makedirs(self.split_save_dir, exist_ok=True)

        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.num_frames = num_frames
        
        all_discovered_samples = [] # Temporary list to hold all samples before splitting

        # Discover samples
        if not os.path.isdir(self.dataset_root_path):
            logging.error(f"Jacquard dataset root path not found or not a directory: {self.dataset_root_path}")
            self.samples = []
            return

        dataset_group_dirs = sorted(glob.glob(os.path.join(self.dataset_root_path, "Jacquard_Dataset_*")))
        if not dataset_group_dirs:
            logging.warning(f"No 'Jacquard_Dataset_*' sub-directories found in {self.dataset_root_path}. Looking for scene_id folders directly under {self.dataset_root_path}.")
            dataset_group_dirs = [self.dataset_root_path]


        for group_dir_path in dataset_group_dirs:
            if not os.path.isdir(group_dir_path):
                continue
            
            scene_dirs = sorted([d for d in glob.glob(os.path.join(group_dir_path, "*")) if os.path.isdir(d)])
            
            if not scene_dirs and group_dir_path == self.dataset_root_path:
                 pass
            elif not scene_dirs:
                 logging.warning(f"No scene directories (e.g., '1a1ec1cfe6...') found in {group_dir_path}")

            for scene_dir_path in scene_dirs:
                rgb_files = sorted(glob.glob(os.path.join(scene_dir_path, "*_RGB.png")))
                
                for rgb_path in rgb_files:
                    base_name = os.path.basename(rgb_path)
                    mask_path = rgb_path.replace("_RGB.png", "_mask.png")
                    grasp_path = rgb_path.replace("_RGB.png", "_grasps.txt")

                    if os.path.exists(mask_path) and os.path.exists(grasp_path):
                        try:
                            # Try to parse frame_prefix (e.g., '0' from '0_sceneid_RGB.png')
                            frame_prefix_str = base_name.split('_')[0]
                            frame_idx_in_scene = int(frame_prefix_str)
                        except (IndexError, ValueError):
                            logging.warning(f"Could not parse frame_prefix from {base_name} in {scene_dir_path}. Using 0 as default.")
                            frame_idx_in_scene = 0 

                        all_discovered_samples.append({
                            "rgb_path": rgb_path,
                            "mask_path": mask_path,
                            "grasp_path": grasp_path,
                            "frame_idx_in_scene": frame_idx_in_scene,
                            "scene_id": os.path.basename(scene_dir_path)
                        })
                    else:
                        logging.warning(f"Skipping {rgb_path}: corresponding mask or grasp file not found.")
        
        if not all_discovered_samples:
            logging.error(f"No Jacquard samples found. Searched in {self.dataset_root_path}.")
            self.samples = []
            return
        
        logging.info(f"Discovered {len(all_discovered_samples)} total Jacquard samples across all scenes.")

        # Perform train/test split based on scene_id
        unique_scene_ids = sorted(list(set(s['scene_id'] for s in all_discovered_samples)))
        
        train_split_file = os.path.join(self.split_save_dir, "jacquard_train_scenes.txt")
        test_split_file = os.path.join(self.split_save_dir, "jacquard_test_scenes.txt")

        train_scene_ids = []
        test_scene_ids = []

        if os.path.exists(train_split_file) and os.path.exists(test_split_file):
            logging.info(f"Loading existing train/test split from {self.split_save_dir}")
            with open(train_split_file, 'r') as f:
                train_scene_ids = [line.strip() for line in f if line.strip()]
            with open(test_split_file, 'r') as f:
                test_scene_ids = [line.strip() for line in f if line.strip()]
            
            # Sanity check: ensure loaded scenes are a subset of discovered scenes
            discovered_set = set(unique_scene_ids)
            if not set(train_scene_ids).issubset(discovered_set) or \
               not set(test_scene_ids).issubset(discovered_set):
                logging.warning("Split files contain scene IDs not found in the current dataset. Re-generating split.")
                train_scene_ids = [] # Force re-generation
                test_scene_ids = []  # Force re-generation
            elif not set(train_scene_ids).isdisjoint(set(test_scene_ids)):
                 logging.warning("Train and test scene IDs in split files are not disjoint. Re-generating split.")
                 train_scene_ids = []
                 test_scene_ids = []

        if not train_scene_ids or not test_scene_ids: # If files didn't exist or forced re-generation
            logging.info(f"Generating new train/test split for Jacquard scenes (Seed: {self.random_seed}).")
            random.seed(self.random_seed)
            shuffled_scene_ids = list(unique_scene_ids) # Make a mutable copy
            random.shuffle(shuffled_scene_ids)
            
            split_point = int(len(shuffled_scene_ids) * self.split_ratio)
            train_scene_ids = shuffled_scene_ids[:split_point]
            test_scene_ids = shuffled_scene_ids[split_point:]

            with open(train_split_file, 'w') as f:
                for scene_id in train_scene_ids:
                    f.write(f"{scene_id}\n")
            with open(test_split_file, 'w') as f:
                for scene_id in test_scene_ids:
                    f.write(f"{scene_id}\n")
            logging.info(f"Saved new train/test split to {self.split_save_dir}")

        target_scene_ids = set()
        if self.mode == 'train':
            target_scene_ids = set(train_scene_ids)
            logging.info(f"Loading Jacquard TRAIN set: {len(target_scene_ids)} scenes.")
        elif self.mode == 'test':
            target_scene_ids = set(test_scene_ids)
            logging.info(f"Loading Jacquard TEST set: {len(target_scene_ids)} scenes.")

        self.samples = [s for s in all_discovered_samples if s['scene_id'] in target_scene_ids]
        
        if not self.samples:
            logging.warning(f"No Jacquard samples found for mode '{self.mode}' with the current split. Check dataset and split files.")
        else:
            logging.info(f"Loaded {len(self.samples)} Jacquard samples for mode '{self.mode}'.")


    def get_video(self, idx):
        """
        For Jacquard, a "video" is a single sample (image, mask, grasps).
        """
        if not 0 <= idx < len(self.samples):
            raise IndexError(f"Index {idx} is out of bounds for JacquardDataset of size {len(self.samples)}.")

        sample_info = self.samples[idx]
        
        rgb_path = sample_info["rgb_path"]
        mask_path = sample_info["mask_path"]
        grasp_path = sample_info["grasp_path"] # grasp_path is collected in __init__

        segment_loader = JacquardSegmentLoader(
            rgb_image_path=rgb_path,
            grasp_file_path=grasp_path,
            mask_file_path=mask_path
        )
        
        # For Jacquard, each item is a single frame.
        # The VOSFrame's frame_idx should be 0 for a single-frame "video".
        # sample_info["frame_idx_in_scene"] is the original prefix from filename, kept for metadata.
        frames = [VOSFrame(frame_idx=0, image_path=rgb_path)]

        # Construct a unique video_name for this sample
        video_name = f"jacquard_{sample_info['scene_id']}_{sample_info['frame_idx_in_scene']}"
        
        video = VOSVideo(video_name=video_name, video_id=idx, frames=frames)
        
        return video, segment_loader

    def __len__(self):
        return len(self.samples)

class GraspNetDataset(VOSRawDataset):
    """
    VOSRawDataset for the GraspNet dataset.
    Each scene is treated as a video.
    """
    def __init__(
        self,
        dataset_root,
        camera_type='kinect',
        num_frames_per_scene=256,
        train_scenes_count=100,
    ):
        """
        Initializes the GraspNetDataset.
        Args:
            dataset_root (str): Path to the GraspNet scenes directory (e.g., '/data/myp/grasp_dataset/scenes').
            camera_type (str): The camera to use, either 'kinect' or 'realsense'.
            num_frames_per_scene (int): The number of frames in each scene sequence.
            train_scenes_count (int): The number of scenes to use for training (e.g., 100 for scenes 0-99).
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.camera_type = camera_type
        self.num_frames_per_scene = num_frames_per_scene
        
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"GraspNet dataset root not found at {self.dataset_root}")

        # Get scene directories for training (e.g., scene_0000 to scene_0099)
        self.scene_paths = []
        for i in range(train_scenes_count):
            scene_name = f"scene_{i:04d}"
            scene_path = os.path.join(self.dataset_root, scene_name)
            if os.path.isdir(scene_path):
                self.scene_paths.append(scene_path)
            else:
                logging.warning(f"Training scene not found and skipped: {scene_path}")
        
        if not self.scene_paths:
            raise FileNotFoundError(f"No training scenes found in {self.dataset_root}")
            
        logging.info(f"Found {len(self.scene_paths)} GraspNet training scenes.")

    def get_video(self, idx):
        """
        Returns a VOSVideo object representing a scene, and a segment loader for its annotations.
        """
        if not 0 <= idx < len(self.scene_paths):
            raise IndexError(f"Index {idx} is out of bounds for GraspNetDataset of size {len(self.scene_paths)}.")

        scene_path = self.scene_paths[idx]
        scene_name = os.path.basename(scene_path)

        # Define paths for the required data directories
        rgb_dir = os.path.join(scene_path, self.camera_type, 'rgb')
        label_dir = os.path.join(scene_path, self.camera_type, 'label')
        rect_dir = os.path.join(scene_path, self.camera_type, 'rect')

        if not all(os.path.isdir(d) for d in [rgb_dir, label_dir, rect_dir]):
            raise FileNotFoundError(f"Required data directories (rgb, label, rect) not found for camera '{self.camera_type}' in scene '{scene_name}'")

        # The GraspNetSegmentLoader will be responsible for loading the actual annotation files (rects, labels) for each frame.
        # I will draft this class for the 'vos_segment_loader.py' file in the next step.
        segment_loader = GraspNetSegmentLoader(
            label_dir=label_dir,
            rect_dir=rect_dir,
        )

        # Create a list of VOSFrame objects for the scene
        frames = []
        for i in range(self.num_frames_per_scene):
            frame_filename = f"{i:04d}.png"
            image_path = os.path.join(rgb_dir, frame_filename)
            if os.path.exists(image_path):
                frames.append(VOSFrame(frame_idx=i, image_path=image_path))
        
        if not frames:
            logging.warning(f"No frames found for scene {scene_name}, although the scene directory exists.")

        video = VOSVideo(video_name=scene_name, video_id=idx, frames=frames)
        
        return video, segment_loader

    def __len__(self):
        return len(self.scene_paths)


class RealworldDataset(VOSRawDataset):
    """
    VOSRawDataset for the Realworld dataset.
    Each zed_xxxx folder is treated as a video.
    """
    def __init__(self, dataset_root):
        """
        Initializes the RealworldDataset.
        Args:
            dataset_root (str): Path to the Realworld dataset root directory.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.video_paths = []

        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Realworld dataset root not found at {self.dataset_root}")

        # Each zed_xxxx folder is a video
        for zed_folder in sorted(os.listdir(self.dataset_root)):
            zed_path = os.path.join(self.dataset_root, zed_folder)
            if os.path.isdir(zed_path) and zed_folder.startswith("zed_"):
                self.video_paths.append(zed_path)
        
        if not self.video_paths:
            logging.warning(f"No video folders (zed_xxxx) found in Realworld dataset at {self.dataset_root}")

        logging.info(f"Found {len(self.video_paths)} Realworld videos.")

    def get_video(self, idx):
        """
        Returns a VOSVideo object representing a zed folder, and a segment loader for its annotations.
        """
        if not 0 <= idx < len(self.video_paths):
            raise IndexError(f"Index {idx} is out of bounds for RealworldDataset of size {len(self.video_paths)}.")

        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path)

        segment_loader = RealworldSegmentLoader(
            video_path=video_path,
        )

        frames = []
        for filename in sorted(os.listdir(video_path)):
            if filename.endswith(".png") and not filename.endswith("_labeled_mask.png"):
                frame_id_str = filename.replace("frame_", "").replace(".png", "")
                try:
                    frame_idx = int(frame_id_str)
                    image_path = os.path.join(video_path, filename)
                    # Check if annotation exists before adding the frame
                    label_path = os.path.join(video_path, f"frame_{frame_id_str}Label.txt")
                    mask_path = os.path.join(video_path, f"frame_{frame_id_str}_labeled_mask.png")
                    if os.path.exists(label_path) and os.path.exists(mask_path):
                         frames.append(VOSFrame(frame_idx=frame_idx, image_path=image_path))
                except ValueError:
                    continue # Skip if frame ID is not a valid integer
        
        if not frames:
            logging.warning(f"No valid annotated frames found for video {video_name}.")

        video = VOSVideo(video_name=video_name, video_id=idx, frames=frames)
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_paths)
