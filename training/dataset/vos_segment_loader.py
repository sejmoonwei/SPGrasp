# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import cv2
from training.grasp_dataset.grasp_OCID import GraspMat
from training.grasp_dataset.grasp_Jacquard import GraspMat_Jacquard
import logging
from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils
except:
    pass


class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])
                    else:
                        annotations_fps = int(data["fps"])
                    assert frames_fps % annotations_fps == 0
                    self.ann_every = frames_fps // annotations_fps
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        assert frame_id % self.ann_every == 0
        rle_mask = self.frame_annots[frame_id // self.ann_every]

        valid_objs_ids = set(range(len(rle_mask)))
        if self.valid_obj_ids is not None:
            # Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # Decode the masks
        raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]
        return segments

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        res = {obj_id: [] for obj_id in range(num_objects)}

        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        video_png_root: the folder contains all the masks stored in png
        """
        self.video_png_root = video_png_root
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        # png_filenames = os.listdir(self.video_png_root)
        png_filenames = [filename for filename in os.listdir(self.video_png_root) if os.path.splitext(filename)[0].isdigit()]
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            frame_id, _ = os.path.splitext(filename)
            self.frame_id_to_png_filename[int(frame_id)] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # load the mask
        masks = PILImage.open(mask_path).convert("P")
        masks = np.array(masks)

        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # remove background (0)

        # convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments #dict1 1280,720  bool

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]
        self.W = tmp_mask.shape[1]
        if self.single_object_mode:
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]  #这里判断是不是encode的形式   如果是mask不需要解码  2258 1500
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0] #2258，1500
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()





class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]  #list 81

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        return self.segments



class OCIDSegmentLoader:
    """
    Loads and manages OCID dataset segments with lazy decoding.
    Combines the functionalities of original OCIDSegmentLoader, LazySegments_OCID, and annotation_OCID.
    """
    def __init__(self, video_label_path, video_frame_path):
        # --- Merged from annotation_OCID.__init__ ---
        self.video_label_path = video_label_path
        self.video_frame_path = video_frame_path
        self.image_id = os.path.basename(video_frame_path)
        self.video_mask_path = video_frame_path.replace('rgb', 'label')
        # self.grasp_poses = [] # Not used by original loader logic being merged
        # self.mask_array = None # Not used by original loader logic being merged
        # --- End merge annotation_OCID.__init__ ---

        # --- Merged from annotation_OCID.get_annots() & gen_annote() ---
        label = GraspMat(self.video_label_path, self.video_mask_path)
        raw_annotations_data = {
            "image": {
                "image_id": self.image_id,
                "width": 640,  # Assuming fixed width, as in original
                "height": 480, # Assuming fixed height, as in original
                "file_name": self.image_id
            },
            "annotations": label.annotations()
        }
        # --- End merge annotation_OCID.get_annots() & gen_annote() ---

        self.frame_annots_list = raw_annotations_data['annotations']

        # --- Merged from LazySegments_OCID ---
        self._rle_segments = {}  # Stores raw RLE data, equivalent to LazySegments_OCID.segments
        self._decoded_cache = {} # Stores decoded masks, equivalent to LazySegments_OCID.cache
        # --- End merge LazySegments_OCID ---

        # Populate _rle_segments
        # Original OCIDSegmentLoader logic for extracting RLE masks
        for i, frame_annot in enumerate(self.frame_annots_list):
            # Assuming frame_annot["segmentation"] is the RLE data
            # and 'i' can serve as the key, similar to how LazySegments was populated.
            self._rle_segments[i] = frame_annot["segmentation"]

    def load(self, frame_idx):
        """
        Returns the loader instance itself, which can be used to access segments.
        The frame_idx is not directly used for selecting RLEs here, as all are loaded
        into _rle_segments during init. This matches SA1BSegmentLoader behavior
        where load() returns the LazySegments instance.
        """
        return self

    # --- Methods merged from LazySegments_OCID ---
    def __getitem__(self, key):
        if key in self._decoded_cache:
            return self._decoded_cache[key]
        
        if key not in self._rle_segments:
            raise KeyError(f"Key {key} not found in RLE segments.")

        rle_data = self._rle_segments[key]
        # Assuming rle_data is a dictionary with "counts" as in original LazySegments_OCID
        # and that rle_data["counts"] is a numpy array suitable for torch.from_numpy
        mask = torch.from_numpy(rle_data["counts"])
        self._decoded_cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self._rle_segments

    def __len__(self):
        return len(self._rle_segments)

    def keys(self):
        return self._rle_segments.keys()


class JacquardSegmentLoader:
    """
    Loads and manages Jacquard dataset segments with lazy decoding,
    mimicking OCIDSegmentLoader's structure.
    Assumes GraspMat will be adapted for Jacquard and provide annotations
    including segmentation data (e.g., {"counts": numpy_mask_array}).
    """
    def __init__(self, rgb_image_path: str, grasp_file_path: str, mask_file_path: str):
        self.rgb_image_path = rgb_image_path
        self.grasp_file_path = grasp_file_path
        self.mask_file_path = mask_file_path # mask_file_path might be used by GraspMat

        self.image_id = os.path.basename(self.rgb_image_path)
        
        # Placeholder for GraspMat call. This will be adapted later.
        # For now, we assume GraspMat is called and label.annotations() provides the necessary structure.
        # If GraspMat is not ready, this part will raise an error or need dummy data.
        
        label_generator = GraspMat_Jacquard(self.grasp_file_path, self.mask_file_path)
        # We expect label_generator.annotations() to return a list of dicts,
        # where each dict is an annotation for an object and contains a "segmentation" field.
        # The "segmentation" field is expected to be a dict with a "counts" key,
        # whose value is a NumPy array of the mask.
        object_annotations_from_graspmat = label_generator.annotations()



        # Store the raw annotations list (list of dicts, one per object)
        self.object_annotations = object_annotations_from_graspmat

        # _segment_data will store the raw segmentation info (e.g., {"counts": np.array}) for each object by its index
        self._segment_data = {}
        # _decoded_cache will store the decoded torch.Tensor masks
        self._decoded_cache = {}

        for i, annotation in enumerate(self.object_annotations):
            if "segmentation" in annotation and isinstance(annotation["segmentation"], dict) and "counts" in annotation["segmentation"]:
                self._segment_data[i] = annotation["segmentation"] # Store the dict like {"counts": np_array}
            else:
                # Log a warning or error if segmentation data is not in the expected format
                # This depends on how critical it is for every annotation to have valid segmentation.
                # For now, we'll skip if not present/correct, or an error could be raised.
                print(f"Warning: Segmentation data for object index {i} in {self.image_id} is missing or not in expected format.")
                # self._segment_data[i] = None # Or handle as an error

    def load(self, frame_idx=None):
        """
        Returns the loader instance itself, which can be used to access segments.
        The frame_idx is ignored, kept for compatibility.
        """
        return self

    def __getitem__(self, key: int): # key is the object index
        if key in self._decoded_cache:
            return self._decoded_cache[key]
        
        if key not in self._segment_data:
            raise KeyError(f"Object index {key} not found in segment data for {self.image_id}.")

        segment_data_for_obj = self._segment_data[key]
        
        # Assuming segment_data_for_obj is a dict like {"counts": numpy_array_mask}
        # and "counts" contains the actual mask data as a NumPy array.
        if not isinstance(segment_data_for_obj, dict) or "counts" not in segment_data_for_obj:
            raise ValueError(f"Segmentation data for object index {key} in {self.image_id} is not in the expected format (dict with 'counts' key). Found: {segment_data_for_obj}")

        mask_numpy_array = segment_data_for_obj["counts"]
        if not isinstance(mask_numpy_array, np.ndarray):
            raise TypeError(f"Expected 'counts' to be a NumPy array for object index {key}, got {type(mask_numpy_array)}.")
            
        mask_tensor = torch.from_numpy(mask_numpy_array).bool() # Ensure it's a boolean tensor
        
        # Expected image dimensions for Jacquard (as per user request)
        # If the mask from GraspMat is not 1024x1024, it might need resizing here or in GraspMat.
        # For now, we assume GraspMat provides masks of the correct final dimensions.
        # if mask_tensor.shape[-2:] != (1024, 1024):
        #     print(f"Warning: Mask for object {key} in {self.image_id} has shape {mask_tensor.shape}, expected (1024,1024).")
            # Potentially add resizing logic if necessary and not handled by GraspMat/transforms

        self._decoded_cache[key] = mask_tensor
        return mask_tensor

    def __contains__(self, key: int):
        return key in self._segment_data

    def __len__(self):
        # Represents the number of objects for which segmentation data was loaded
        return len(self._segment_data)

    def keys(self):
        # Returns the indices of the objects
        return self._segment_data.keys()

    def get_image_info(self):
        """Returns basic image information."""
        return {
            "image_id": self.image_id,
            "width": 1024, # Fixed as per requirement
            "height": 1024, # Fixed as per requirement
            "file_name": os.path.basename(self.rgb_image_path)
        }

    def get_object_annotations(self):
        """Returns the list of raw object annotations obtained from GraspMat."""
        return self.object_annotations

def _compute_grasp_rectangles_from_graspnet_format(graspnet_npy_data):
    """
    Converts GraspNet's .npy file format to a list of quadrilaterals.
    Directly adapted from `compute_grasp_rectangles` in the reference script.
    """
    rectangles = []
    for grasp in graspnet_npy_data:
        xcenter, ycenter, xright, yright, height, _, _ = grasp
        width = 2 * np.sqrt((xright - xcenter)**2 + (yright - ycenter)**2)
        angle = np.arctan2(yright - ycenter, xright - xcenter)
        dx, dy = width / 2, height / 2
        corners = [(-dx, dy), (-dx, -dy), (dx, -dy), (dx, dy)]
        rotated_corners = [
            (xcenter + x * np.cos(angle) - y * np.sin(angle),
             ycenter + x * np.sin(angle) + y * np.cos(angle))
            for (x, y) in corners
        ]
        rectangles.append(rotated_corners)
    return rectangles

def _generate_global_grasp_map_from_rectangles(rectangles, image_shape):
    """
    Generates a 3-channel grasp map (pos, angle, width) from a list of quadrilaterals.
    Adapted from `gen_mat` in the reference script.
    """
    h, w = image_shape
    grasp_map = np.zeros((3, h, w), dtype=np.float32)
    for rect in rectangles:
        center_x, center_y = np.mean([p[0] for p in rect]), np.mean([p[1] for p in rect])
        p1, p2, p3, _ = rect
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        width = np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        
        row, col = int(center_y), int(center_x)
        if 0 <= row < h and 0 <= col < w:
            size = 3
            start_row, end_row = max(0, row - size), min(h, row + size + 1)
            start_col, end_col = max(0, col - size), min(w, col + size + 1)
            grasp_map[0, start_row:end_row, start_col:end_col] = 1.0
            grasp_map[1, start_row:end_row, start_col:end_col] = angle
            grasp_map[2, start_row:end_row, start_col:end_col] = width
    return grasp_map

class GraspNetSegmentLoader:
    """
    Refactored SegmentLoader for the GraspNet dataset.
    It processes all objects in the original image resolution.
    """
    def __init__(self, label_dir, rect_dir):
        self.label_dir = label_dir
        self.rect_dir = rect_dir
        self._cache = {}

    def load(self, frame_id, obj_ids=None):
        if frame_id in self._cache:
            return self._cache[frame_id]

        label_path = os.path.join(self.label_dir, f"{frame_id:04d}.png")
        rect_path = os.path.join(self.rect_dir, f"{frame_id:04d}.npy")

        if not (os.path.exists(label_path) and os.path.exists(rect_path)):
            logging.warning(f"Annotations not found for frame {frame_id}")
            return {}

        try:
            semantic_mask = np.array(PILImage.open(label_path))
            graspnet_npy_data = np.load(rect_path)
        except Exception as e:
            logging.error(f"Failed to load annotation files for frame {frame_id}: {e}")
            return {}

        grasp_rectangles = _compute_grasp_rectangles_from_graspnet_format(graspnet_npy_data)
        global_grasp_map = _generate_global_grasp_map_from_rectangles(grasp_rectangles, semantic_mask.shape)

        unique_obj_ids = np.unique(semantic_mask)
        unique_obj_ids = [oid for oid in unique_obj_ids if oid != 0]

        if obj_ids is not None:
            unique_obj_ids = [oid for oid in unique_obj_ids if oid in obj_ids]

        segments = {}
        for obj_id in unique_obj_ids:
            instance_mask = (semantic_mask == obj_id)
            object_grasp_map = np.where(np.expand_dims(instance_mask, axis=0), global_grasp_map, 0)

            if np.max(object_grasp_map[0]) == 0: # Check if any grasp confidence exists for this object
                continue

            decoded_grasp_map = GraspMat.decode(object_grasp_map) #
            semantic_channel = np.expand_dims(instance_mask.astype(np.float32), axis=0)
            final_segment = np.concatenate((decoded_grasp_map, semantic_channel), axis=0)
            segments[obj_id] = torch.from_numpy(final_segment)

        # Debugging check for tensor shapes before returning
        expected_shape = (5, 720, 1280)
        for obj_id, tensor in segments.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for object ID {obj_id} in frame {frame_id}. "
                    f"Expected shape {expected_shape}, but got {tensor.shape}. "
                    "You can now inspect the variables."
                )

        self._cache[frame_id] = segments
        return segments

class RealworldSegmentLoader:
    """
    SegmentLoader for the Realworld dataset.
    It processes all objects in the original image resolution for a given frame in a video.
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self._cache = {}

    def load(self, frame_id, obj_ids=None):
        if frame_id in self._cache:
            return self._cache[frame_id]

        frame_id_str = f"{frame_id:06d}"
        label_path = os.path.join(self.video_path, f"frame_{frame_id_str}Label.txt")
        mask_path = os.path.join(self.video_path, f"frame_{frame_id_str}_labeled_mask.png")

        if not (os.path.exists(label_path) and os.path.exists(mask_path)):
            logging.warning(f"Annotations not found for frame {frame_id} in {self.video_path}")
            return {}

        try:
            instance_mask = np.array(PILImage.open(mask_path))
            if instance_mask.ndim == 3:
                instance_mask = instance_mask[:, :, 0]

            with open(label_path, 'r') as f:
                grasp_lines = [line.strip().split() for line in f if line.strip()]
            grasps = np.array(grasp_lines, dtype=np.float32)

        except Exception as e:
            logging.error(f"Failed to load annotation files for frame {frame_id}: {e}")
            return {}

        h, w = instance_mask.shape
        segments = {}
        
        unique_obj_ids = np.unique(instance_mask)
        unique_obj_ids = [oid for oid in unique_obj_ids if oid != 0]

        if obj_ids is not None:
            unique_obj_ids = [oid for oid in unique_obj_ids if oid in obj_ids]

        for obj_id in unique_obj_ids:
            current_instance_mask = (instance_mask == obj_id)
            
            obj_grasps = []
            for grasp in grasps:
                row, col = int(grasp[0]), int(grasp[1])
                if 0 <= row < h and 0 <= col < w and current_instance_mask[row, col]:
                    obj_grasps.append(grasp)
            
            if not obj_grasps:
                continue

            obj_grasp_map = np.zeros((4, h, w), dtype=np.float32)
            
            for grasp in obj_grasps:
                row, col, angle_rad, _, width = grasp
                row, col = int(row), int(col)
                
                size = 5
                start_row, end_row = max(0, row - size), min(h, row + size + 1)
                start_col, end_col = max(0, col - size), min(w, col + size + 1)
                
                obj_grasp_map[0, start_row:end_row, start_col:end_col] = 1.0
                obj_grasp_map[1, start_row:end_row, start_col:end_col] = np.cos(2 * angle_rad)
                obj_grasp_map[2, start_row:end_row, start_col:end_col] = np.sin(2 * angle_rad)
                # Normalize width to be in a range suitable for BCE loss, similar to other datasets.
                # A common practice is to normalize by a factor related to max grasp width.
                # Using 150.0 as a reasonable normalization factor.
                obj_grasp_map[3, start_row:end_row, start_col:end_col] = width / 100.0

            semantic_channel = np.expand_dims(current_instance_mask.astype(np.float32), axis=0)
            final_segment = np.concatenate((obj_grasp_map, semantic_channel), axis=0)
            
            segments[int(obj_id)] = torch.from_numpy(final_segment)

        self._cache[frame_id] = segments
        return segments
