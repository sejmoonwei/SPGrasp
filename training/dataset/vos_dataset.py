# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy
import os
import matplotlib.pyplot as plt

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target= True, 
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            # sample a video
            video, segment_loader = self.video_dataset.get_video(idx)
            # sample frames and object indices to be used in a datapoint
            sampled_frms_and_objs = self.sampler.sample(
                video, segment_loader, epoch=self.curr_epoch
            )
            break  # Succesfully loaded video


        #fix complete
        datapoint = self.construct(video, sampled_frms_and_objs, segment_loader)  #结构不一样 但是不在这里使用 可以过 wid 0.4 0.5 0.6 

        
        for transform in self._transforms:  
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):  #可以在这里面做变换，因为传进来的是路径，出去的是mask
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = segment_loader.load(frame.frame_idx)   # dict 56 每个元素包含尺寸和rle
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None  #here
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    if segments[obj_id].dtype == torch.uint8:
                        segment = segments[obj_id] #fix 2258，1500  segments 在这里才变为 mask  train1只有0-1量
                        #debug
                        raise ValueError(
                            "You can now inspect the variables.")
                    else:
                        segment = segments[obj_id] #.to(torch.uint8) #need fix
                        if len(segment.shape) != 3:
                            raise ValueError(
                            "You can now inspect the variables.")
                     #train2 float64  train1  uint8
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(5, h, w, dtype=torch.uint8) #fixed

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        datapoint = self._get_datapoint(idx)
        # Visualize the datapoint before returning it for debugging purposes.
        vis_datapoint(datapoint)
        return datapoint

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)


# Denormalization values for visualization
# These are standard ImageNet values, matching the config
VIS_MEAN = np.array([0.485, 0.456, 0.406])
VIS_STD = np.array([0.229, 0.224, 0.225])

def vis_datapoint(datapoint):
    """
    Visualizes a datapoint by saving the frame image and its corresponding
    label channels as heatmaps for each object.

    Args:
        datapoint (VideoDatapoint): The datapoint to visualize.
    """
    # Use a try-except block to avoid crashing the training if visualization fails
    try:
        output_dir = os.path.join("output", "vos_dataset_vis", str(datapoint.video_id))
        os.makedirs(output_dir, exist_ok=True)

        channel_names = ["pos", "ang_cos", "ang_sin", "width", "semantic"]

        for frame_idx, frame in enumerate(datapoint.frames):
            # --- 1. Prepare Image ---
            img_tensor = frame.data.cpu().clone()  # (C, H, W)

            # De-normalize for visualization
            img_vis = img_tensor.numpy().transpose(1, 2, 0)  # (H, W, C)
            img_vis = (img_vis * VIS_STD) + VIS_MEAN
            img_vis = np.clip(img_vis, 0, 1)

            # --- 2. Iterate over objects in the frame ---
            if not frame.objects:
                continue

            for obj in frame.objects:
                # --- Prepare Labels for the current object ---
                label_tensor = obj.segment.cpu().clone().float()

                if label_tensor.ndim != 3 or label_tensor.shape[0] != len(
                    channel_names
                ):
                    logging.warning(
                        f"Skipping visualization for obj {obj.object_id} in frame {obj.frame_index} due to unexpected label shape: {label_tensor.shape}"
                    )
                    continue

                # --- 3. Create and Save Visualization ---
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(
                    f"Video: {datapoint.video_id}, Frame: {obj.frame_index}, Object ID: {obj.object_id}",
                    fontsize=16,
                )

                # Plot original image
                axes[0, 0].imshow(img_vis)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis("off")

                # Plot label channels as heatmaps
                for i, ax in enumerate(axes.flat[1:]):
                    if i < len(channel_names):
                        channel_data = label_tensor[i].numpy()
                        im = ax.imshow(channel_data, cmap="viridis")
                        ax.set_title(f"Label: {channel_names[i]}")
                        fig.colorbar(im, ax=ax)
                    ax.axis("off")

                # Save the figure with a unique name for the object
                save_path = os.path.join(
                    output_dir, f"frame_{obj.frame_index:04d}_obj_{obj.object_id}.png"
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_path)
                plt.close(fig)
                logging.info(f"Saved VOS visualization to {save_path}")
    except Exception as e:
        logging.error(
            f"Failed to visualize datapoint for video {datapoint.video_id}. Error: {e}"
        )
