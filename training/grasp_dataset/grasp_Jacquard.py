import numpy as np
import cv2
import math
import os # Added for os.path.basename and os.getenv
# import torch # Not directly used in GraspMat, but often by consumers
# import pycocotools.mask as maskUtils # Not used in the new Jacquard GraspMat logic

# Removed standalone helper functions if they are integrated or replaced by class methods.

class GraspMat_Jacquard:
    def __init__(self, label_file: str, ins_mask_path: str, visualize_on_init: bool = False):
        """
        Initializes GraspMat for Jacquard dataset.
        Args:
            label_file (str): Path to the Jacquard grasp file (*_grasps.txt).
            ins_mask_path (str): Path to the instance mask file (*_mask.png).
            visualize_on_init (bool): If True, visualizes grasps upon initialization.
        """
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
        if not os.path.exists(ins_mask_path):
            raise FileNotFoundError(f"Instance mask file not found: {ins_mask_path}")

        self.label_file = label_file
        self.mask_file = ins_mask_path
        self.rgb_file = self.label_file.replace('_grasps.txt', '_RGB.png')
        if not os.path.exists(self.rgb_file):
            # Attempt to find RGB if replacement didn't work (e.g. different naming or case)
            rgb_candidate_base = os.path.basename(self.label_file).split('_grasps.txt')[0]
            potential_rgb_name = f"{rgb_candidate_base}_RGB.png"
            potential_rgb_path = os.path.join(os.path.dirname(self.label_file), potential_rgb_name)
            if os.path.exists(potential_rgb_path):
                self.rgb_file = potential_rgb_path
            else:
                # Fallback or warning if RGB is strictly needed for something other than visualization
                print(f"Warning: RGB file {self.rgb_file} (inferred) or {potential_rgb_path} not found. Visualization might fail.")
                # self.rgb_file = None # Or handle appropriately

        self.label_mat = self._gen_label_mat()
        self.mask_array = self._gen_mask_array() # This loads and ensures mask is 1024x1024
        self.Annotations = self._create_annotations()

        if visualize_on_init or os.getenv("VISUALIZE_JACQUARD_GRASPS", "0") == "1":
            self.visualize()

    def _gen_label_mat(self):
        """
        Generates the label matrix (position, angle, width) from Jacquard grasp file.
        Output dimensions are (3, 1024, 1024).
        """
        label_mat = np.zeros((3, 1024, 1024), dtype=np.float64)
        # Channel 0: Position map (grasp center)
        # Channel 1: Angle map (radians, normalized to [0, pi))
        # Channel 2: Width map (pixels)

        center_marker_size = 5 # Radius of the square area to mark around the center

        try:
            with open(self.label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if not parts or len(parts) < 4: # Need at least x, y, angle, width
                        print(f"Warning: Skipping malformed line in {self.label_file}: {line.strip()}")
                        continue
                    
                    try:
                        col_center = float(parts[0])
                        row_center = float(parts[1])
                        theta_degrees = float(parts[2])
                        width_pixels  = float(parts[3])
                    except ValueError:
                        print(f"Warning: Could not parse numeric values in line: {line.strip()} from {self.label_file}")
                        continue

                    r_idx, c_idx = int(round(row_center)), int(round(col_center))

                    r_start = max(0, r_idx - center_marker_size)
                    r_end = min(1024, r_idx + center_marker_size + 1)
                    c_start = max(0, c_idx - center_marker_size)
                    c_end = min(1024, c_idx + center_marker_size + 1)

                    # Channel 0: Position map
                    label_mat[0, r_start:r_end, c_start:c_end] = 1.0
                    
                    # Channel 1: Angle map
                    angle_rad = -theta_degrees / 180.0 * np.pi
                    while angle_rad < 0:
                        angle_rad += math.pi
                    while angle_rad >= math.pi:
                        angle_rad -= math.pi
                    label_mat[1, r_start:r_end, c_start:c_end] = angle_rad
                    
                    # Channel 2: Width map
                    label_mat[2, r_start:r_end, c_start:c_end] = width_pixels

        except FileNotFoundError:
            print(f"Error: Label file not found at {self.label_file}")
            return label_mat
        except Exception as e:
            print(f"Error processing label file {self.label_file}: {e}")
            return label_mat

        return label_mat

    def _decode_label_mat_for_segmentation(self, label_mat_3channel):
        """
        Decodes a 3-channel label_mat (pos, angle, width) into a 4-channel
        representation (pos, cos(2*angle), sin(2*angle), normalized_width)
        for consistency with OCID's segmentation format.
        """
        if label_mat_3channel.shape[0] != 3:
            raise ValueError(f"Input label_mat_3channel must have 3 channels, got {label_mat_3channel.shape[0]}")

        h, w = label_mat_3channel.shape[1:]
        decoded_mat = np.zeros((4, h, w), dtype=np.float64)

        # Channel 0: Position confidence
        decoded_mat[0] = label_mat_3channel[0]

        # Channel 1 & 2: cos(2*angle) and sin(2*angle)
        angle_rad_map = label_mat_3channel[1]
        decoded_mat[1] = np.cos(2 * angle_rad_map)
        decoded_mat[2] = np.sin(2 * angle_rad_map)
        # Zero out cos/sin where there's no grasp point, to avoid cos(0)=1, sin(0)=0 for non-grasp areas
        no_grasp_mask = (label_mat_3channel[0] == 0)
        decoded_mat[1][no_grasp_mask] = 0
        decoded_mat[2][no_grasp_mask] = 0


        # Channel 3: Normalized width
        # Normalize by image width (1024), similar to OCID's width / 100.0 if its width was in mm.
        # This brings width to a [0, ~1] range.
        decoded_mat[3] = label_mat_3channel[2] / 1024.0
        # Ensure width is also zeroed out where there's no grasp
        decoded_mat[3][no_grasp_mask] = 0


        return decoded_mat

    def _gen_mask_array(self):
        """Loads and prepares the instance mask from self.mask_file."""
        mask = cv2.imread(self.mask_file, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Mask file not found or unreadable: {self.mask_file}. Returning a full zero mask.")
            return np.zeros((1024, 1024), dtype=np.uint8) # Default empty mask

        # Ensure mask is 2D (e.g. if it's loaded as 3-channel grayscale)
        if len(mask.shape) == 3 and mask.shape[2] in [3, 4]:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY if mask.shape[2]==3 else cv2.COLOR_BGRA2GRAY)
            print(f"Warning: Mask file {self.mask_file} was multi-channel, converted to grayscale.")

        if mask.shape[0] != 1024 or mask.shape[1] != 1024:
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return mask # Should be (1024, 1024)

    def _create_annotations(self):
        """
        Creates annotations list in the format expected by JacquardSegmentLoader,
        combining grasp information (self.label_mat) with instance masks (self.mask_array).
        """
        annotations_list = []
        unique_instance_ids = np.unique(self.mask_array)
        # Assuming 0 is background, filter it out.
        # Other non-zero values are instance IDs.
        unique_instance_ids = [uid for uid in unique_instance_ids if uid != 0]

        if not unique_instance_ids:
            # If mask is empty or only background, but grasps might exist (apply to whole scene)
            if np.any(self.label_mat[0] > 0): # Check if any grasp points exist
                print(f"Warning: No distinct instances in mask {self.mask_file}, but grasps exist. Creating one annotation for the whole scene.")
                # Use the full label_mat and a full "true" mask for this single annotation
                full_scene_mask_np = np.ones((1024, 1024), dtype=bool)
                
                # Decode the 3-channel label_mat to 4-channel for segmentation format
                decoded_label_mat = self._decode_label_mat_for_segmentation(self.label_mat) # (4, 1024, 1024)
                
                # Data for 'counts': 4 channels from decoded_label_mat + 1 channel for instance mask
                combined_data = np.concatenate(
                    (decoded_label_mat, np.expand_dims(full_scene_mask_np.astype(np.float64), axis=0)),
                    axis=0
                ) # Shape (5, 1024, 1024)
                
                segmentation_dict = {'counts': combined_data}
                # Bbox and point for full scene
                bbox_flat = [0, 0, 1024, 1024]
                point_coords_list = [[512.0, 512.0]] # Center point

                annotations_list.append({
                    'bbox': bbox_flat,
                    'area': float(np.sum(full_scene_mask_np)),
                    'segmentation': segmentation_dict,
                    'point_coords': point_coords_list,
                })
            return annotations_list # Return empty or single scene annotation

        for instance_id_val in unique_instance_ids:
            instance_mask_np = (self.mask_array == instance_id_val) # Binary mask (1024, 1024) for current instance

            # Filter the global label_mat by this instance's mask
            # Filter the global 3-channel label_mat by this instance's mask
            instance_specific_label_mat = np.zeros_like(self.label_mat) # Still (3, 1024, 1024)
            for ch in range(self.label_mat.shape[0]): # 3 channels
                instance_specific_label_mat[ch] = np.where(instance_mask_np, self.label_mat[ch], 0)

            # Decode the instance-specific 3-channel label_mat to 4-channel
            decoded_instance_label_mat = self._decode_label_mat_for_segmentation(instance_specific_label_mat) # (4, 1024, 1024)
            
            # 'counts' should be (5, H, W): 4 from decoded_instance_label_mat + 1 for instance_mask_np
            combined_data_for_counts = np.concatenate(
                (decoded_instance_label_mat, np.expand_dims(instance_mask_np.astype(np.float64), axis=0)),
                axis=0
            )

            segmentation_dict = {'counts': combined_data_for_counts}
            
            # Derive bbox and point from the instance_mask_np
            # Using the static methods (assuming they are defined in this class or accessible)
            if np.any(instance_mask_np): # Ensure mask is not empty before sampling points/bbox
                point = GraspMat_Jacquard.sample_point(instance_mask_np)
                bbox_corners = GraspMat_Jacquard.sample_box(instance_mask_np)
                bbox_flat = [coord for pt_pair in bbox_corners for coord in pt_pair] # Flatten
                point_coords_list = [[float(p) for p in point]] # Ensure float for JSON if needed
            else: # Should not happen if instance_id_val came from unique non-zero values
                bbox_flat = [0,0,0,0]
                point_coords_list = [[0.0,0.0]]


            annotations_list.append({
                'bbox': bbox_flat,
                'area': float(np.sum(instance_mask_np)),
                'segmentation': segmentation_dict,
                'point_coords': point_coords_list,
                # 'id': int(instance_id_val), # Optionally include the instance ID from mask
            })
        return annotations_list

    def annotations(self):
        return self.Annotations

    def visualize(self, show_opencv_window=True):
        """Visualizes the grasps from self.label_file on the RGB image."""
        if not hasattr(self, 'rgb_file') or not self.rgb_file or not os.path.exists(self.rgb_file):
            print(f"RGB file not available or not found ({getattr(self, 'rgb_file', 'N/A')}), cannot visualize.")
            return None
        
        img = cv2.imread(self.rgb_file)
        if img is None:
            print(f"Failed to load RGB image: {self.rgb_file}")
            return None

        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

        try:
            with open(self.label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if not parts or len(parts) < 4:
                        continue
                    try:
                        col_center = float(parts[0])
                        row_center = float(parts[1])
                        theta_degrees = float(parts[2])
                        width_pixels = float(parts[3])
                    except ValueError:
                        continue

                    # Convert to int for drawing
                    x_c, y_c = int(round(col_center)), int(round(row_center))
                    w_half = width_pixels / 2.0

                    # Angle for drawing: Jacquard angles are typically relative to image x-axis
                    # Positive angle often means counter-clockwise rotation from positive x-axis.
                    # A common way to draw a gripper: line between (x_c - w_half*sin, y_c + w_half*cos) and (x_c + w_half*sin, y_c - w_half*cos)
                    # where angle is in radians.
                    angle_rad_vis = theta_degrees * np.pi / 180.0 # Direct conversion for visualization
                                                                # No negation as in label_mat, if theta is already defined as desired for vis.
                                                                # Or use the same angle as in label_mat for consistency:
                    # angle_rad_vis = -theta_degrees / 180.0 * np.pi
                    # while angle_rad_vis >= math.pi: angle_rad_vis -= math.pi
                    # while angle_rad_vis < 0: angle_rad_vis += math.pi


                    # Gripper line endpoints
                    # p1_x = x_c + w_half * math.sin(angle_rad_vis) # Corrected: should be cos for x-offset of perpendicular, sin for y-offset of perpendicular
                    # p1_y = y_c - w_half * math.cos(angle_rad_vis)
                    # p2_x = x_c - w_half * math.sin(angle_rad_vis)
                    # p2_y = y_c + w_half * math.cos(angle_rad_vis)
                    
                    # Alternative: draw line along angle_rad_vis, then perpendiculars for width
                    # Let angle_rad_vis be the orientation of the gripper opening (axis of approach)
                    # Gripper fingers are perpendicular to this.
                    # If theta_degrees is the angle of the gripper's major axis (line connecting fingers):
                    dx_gripper = w_half * math.cos(angle_rad_vis)
                    dy_gripper = w_half * math.sin(angle_rad_vis)

                    pt1_x, pt1_y = int(x_c - dx_gripper), int(y_c - dy_gripper)
                    pt2_x, pt2_y = int(x_c + dx_gripper), int(y_c + dy_gripper)
                    
                    cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2) # Line representing gripper width/orientation
                    cv2.circle(img, (x_c, y_c), 3, (255, 0, 0), -1) # Center point
                    
                    # Optionally, draw small lines for the gripper jaws (perpendicular to the main line)
                    jaw_len = max(5.0, width_pixels * 0.15) # Length of jaw lines
                    angle_jaw = angle_rad_vis + math.pi / 2.0 # Perpendicular
                    
                    dx_jaw = jaw_len * math.cos(angle_jaw)
                    dy_jaw = jaw_len * math.sin(angle_jaw)

                    cv2.line(img, (pt1_x, pt1_y), (int(pt1_x + dx_jaw), int(pt1_y + dy_jaw)), (0, 0, 255), 2)
                    cv2.line(img, (pt1_x, pt1_y), (int(pt1_x - dx_jaw), int(pt1_y - dy_jaw)), (0, 0, 255), 2)
                    cv2.line(img, (pt2_x, pt2_y), (int(pt2_x + dx_jaw), int(pt2_y + dy_jaw)), (0, 0, 255), 2)
                    cv2.line(img, (pt2_x, pt2_y), (int(pt2_x - dx_jaw), int(pt2_y - dy_jaw)), (0, 0, 255), 2)


        except FileNotFoundError:
            print(f"Error: Label file not found at {self.label_file} during visualization.")
            return img # Return original image if labels can't be read
        except Exception as e:
            print(f"Error during visualization: {e}")
            return img


        if show_opencv_window:
            # Save the image instead of showing it
            if hasattr(self, 'rgb_file') and self.rgb_file and img is not None:
                rgb_file_dir = os.path.dirname(self.rgb_file)
                # Create a subdirectory for visualizations if it doesn't exist
                vis_output_dir = os.path.join(rgb_file_dir, "jacquard_visualized_grasps")
                os.makedirs(vis_output_dir, exist_ok=True)

                rgb_file_basename, _ = os.path.splitext(os.path.basename(self.rgb_file))
                save_filename = f"{rgb_file_basename}_visualized.png"
                save_path = os.path.join(vis_output_dir, save_filename)
                
                cv2.imwrite(save_path, img)
                print(f"Visualization saved to: {save_path}")
            else:
                print("Warning: RGB file path not available or image not loaded, cannot save visualization.")
        return img

    # Static methods from OCID GraspMat, can be kept if useful
    @staticmethod
    def sample_point(positions_mask: np.ndarray) -> tuple:
        coords = np.argwhere(positions_mask)
        if coords.size == 0:
            # Return a default point (e.g. center of image) if mask is empty
            # This case should be handled carefully based on how an empty mask is treated.
            # For now, raise error as per original, or return a sensible default.
            # print("Warning: Mask for sample_point is empty. Returning default point (0,0).")
            # return (0,0) # Or image center e.g. (512,512) for 1024x1024
            raise ValueError("Mask for sample_point is empty.")
        idx = np.random.choice(len(coords))
        return tuple(coords[idx]) # Returns (row, col)

    @staticmethod
    def sample_box(positions_mask: np.ndarray) -> tuple:
        coords = np.argwhere(positions_mask)
        if coords.size == 0:
            # print("Warning: Mask for sample_box is empty. Returning default bbox (0,0,0,0).")
            # return ((0,0),(0,0))
            raise ValueError("Mask for sample_box is empty.")
        rows = coords[:, 0]
        cols = coords[:, 1]
        return ((int(rows.min()), int(cols.min())), (int(rows.max()), int(cols.max())))

    # The _decode and decode methods from OCID are specific to its 3-channel label_mat
    # and angle representation. They are likely not needed for Jacquard's 4-channel label_mat
    # where angle and width are already directly stored.
    # If a similar "decoding" step is needed for Jacquard before consumption by the model,
    # it would be a new method specific to Jacquard's label_mat format.
    # For now, removing them to avoid confusion, as JacquardSegmentLoader expects 'counts'
    # to be the (5, H, W) combined data.

# Example usage (for testing this script directly)
if __name__ == '__main__':
    # This part is for testing and will not run when imported as a module.
    # Create dummy files based on the Jacquard structure provided by the user.
    print("Running GraspMat Jacquard direct test...")
    base_dir = "dummy_jacquard_data"
    scene_id = "test_scene_01"
    sample_prefix = "0"
    
    os.makedirs(os.path.join(base_dir, scene_id), exist_ok=True)
    
    rgb_file = os.path.join(base_dir, scene_id, f"{sample_prefix}_{scene_id}_RGB.png")
    mask_file = os.path.join(base_dir, scene_id, f"{sample_prefix}_{scene_id}_mask.png")
    grasp_file = os.path.join(base_dir, scene_id, f"{sample_prefix}_{scene_id}_grasps.txt")

    # Create a dummy RGB image (1024x1024)
    dummy_rgb = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite(rgb_file, dummy_rgb)

    # Create a dummy mask image (1024x1024) with one instance (ID=1)
    dummy_mask = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (200, 200), (800, 800), 1, -1) # Instance 1
    cv2.imwrite(mask_file, dummy_mask)

    # Create a dummy grasp file
    with open(grasp_file, 'w') as f:
        # col;row;theta_degrees;width_px;height_px (height is optional for our processing)
        f.write("512.0;512.0;0.0;100.0;50.0\n")  # Grasp 1: center, horizontal, width 100
        f.write("300.0;300.0;45.0;80.0;40.0\n")   # Grasp 2: at (300,300), 45 deg, width 80
        f.write("700.0;600.0;-30.0;120.0;60.0\n") # Grasp 3

    print(f"Dummy files created in {os.path.join(base_dir, scene_id)}")
    print(f"RGB: {rgb_file}")
    print(f"Mask: {mask_file}")
    print(f"Grasp: {grasp_file}")

    try:
        # Test GraspMat instantiation and visualization
        # Pass visualize_on_init=True or set environment variable VISUALIZE_JACQUARD_GRASPS=1
        grasp_handler = GraspMat_Jacquard(grasp_file, mask_file, visualize_on_init=True)
        
        print(f"\nGenerated label_mat shape: {grasp_handler.label_mat.shape}")
        print(f"Number of annotations: {len(grasp_handler.Annotations)}")
        if grasp_handler.Annotations:
            print(f"First annotation segmentation 'counts' shape: {grasp_handler.Annotations[0]['segmentation']['counts'].shape}")
            print(f"Annotation keys: {grasp_handler.Annotations[0].keys()}")
            print(f"Sample point for first annotation: {grasp_handler.Annotations[0]['point_coords']}")
            print(f"Bbox for first annotation: {grasp_handler.Annotations[0]['bbox']}")


        # To test visualization separately:
        # grasp_handler.visualize()

    except Exception as e:
        print(f"Error during GraspMat test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files (optional)
        # import shutil
        # if os.path.exists(base_dir):
        #     shutil.rmtree(base_dir)
        # print(f"Cleaned up dummy data directory: {base_dir}")
        pass

