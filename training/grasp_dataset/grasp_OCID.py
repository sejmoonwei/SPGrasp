import mmcv
import numpy as np
import cv2
import math
import torch
import scipy.io as scio
# from mmcv.ops.roi_align import roi_align
import random
import pycocotools.mask as maskUtils

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasp1(img, grasp):
    """
    绘制抓取标签
        grasp: [row, col, angle, width]
    :return:
    """

    row, col = int(grasp[0]), int(grasp[1])
    cv2.circle(img, (int(grasp[1]), int(grasp[0])), 2, (0, 255, 0), -1)
    angle = grasp[2]   # 弧度
    width = grasp[3] / 2

    k = math.tan(angle)

    if k == 0:
        dx = width
        dy = 0
    else:
        dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
        dy = k * dx

    if angle < math.pi:
        cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
    else:
        cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)


    return img

# 函数计算中心点
def calculate_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return (center_x, center_y)

# 函数计算长度（假设第一个和第二个点为长边）
def calculate_length(points):
    p1, p2 = points[1], points[2]
    length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length

# 函数计算抓取方向与y轴的夹角
def calculate_angle(points):
    p1, p2 = points[0], points[1]
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.atan2(delta_y, delta_x)
    # 将角度转换为与y轴的夹角
    angle_with_y = angle - (math.pi / 2)
    # 将角度转换为度数
    angle_with_y_degrees = math.degrees(angle_with_y)
    return angle_with_y_degrees


def get_grasp(grasp_rects):
    # 收集所有抓取表示
    grasp_representations = []
    for rect in grasp_rects:
        center = calculate_center(rect)
        length = calculate_length(rect)
        angle = calculate_angle(rect)
        grasp_representations.append((center[0], center[1], angle, length))

    return grasp_representations

def gen_mat(label_file):
    with open(label_file, "r") as f:
        points_list = []
        boxes_list = []
        for count, line in enumerate(f):
            line = line.rstrip()
            [x, y] = line.split(' ')

            x = float(x)
            y = float(y)

            pt = (x, y)
            points_list.append(pt)

            if len(points_list) == 4:
                boxes_list.append(points_list) #list [    [(),(),(),()]     ,    ]
                points_list = []

    # get pos, cls, nagle, width
    grasp_label = get_grasp(boxes_list)


    label_mat = np.zeros((3, 480, 640), dtype=np.float64) #pos cls angle width

    for label in grasp_label:
        row = int(float(label[1]))
        col = int(float(label[0]))

        size = 5
        startx = max(0,row-size)
        endx = min(480, row+size+1)
        starty = max(0,col-size)
        endy = min(640, col+size+1)
        label_mat[0,startx:endx,starty:endy] = 1  # 设置抓取点

        theta = float(label[2])  #
        angle = - theta / 180.0 * np.pi
        if angle < -3.14 or angle > 6.28:
            raise ValueError('invalid angle:{}'.format(angle))
        elif angle < 0:
            angle += 3.14
        elif angle > 3.14:
            angle -= 3.14


        label_mat[1,startx:endx,starty:endy] = angle  # 设置抓取点 #same pos with multi angle   not considered yet

        # label_mat[2, row, col] = angle
        label_mat[2,startx:endx,starty:endy] = float(label[-1])   # 设置抓取宽度
    # 0:position  1:angle 2:w
    return label_mat


class GraspMat:
    """

    生成总mask
    物体计数
    每个物体生成一组mask
    每个物体采集一个点

    """
    def __init__(self, file, ins_mask_path):
        #(4,640,480)
        grasp = gen_mat(file) #3,480,640
        all_instance_msk = cv2.imread(ins_mask_path, cv2.IMREAD_UNCHANGED) #（480，640）
        if all_instance_msk is None:
            print(f"Warning: Mask file not found or unreadable: {ins_mask_path}")
            all_instance_msk = np.zeros((480, 640), dtype=np.uint8) # Default empty mask

        unique_mask_values = np.unique(all_instance_msk)
        
        # Initial filtering: remove 0, 1, 2
        potential_instance_ids = [val for val in unique_mask_values if val not in [0, 1, 2]]

        # If, after removing 0, 1, 2, no other values remain,
        # AND 2 was present in the original unique values, then consider 2 as an instance.
        if not potential_instance_ids and 2 in unique_mask_values:
            instance_ids_to_process = [2]
        elif not potential_instance_ids and 2 not in unique_mask_values: # Only 0 or 0,1 present
            instance_ids_to_process = []
        else: # Values other than 0,1,2 exist, so they are the instances
            instance_ids_to_process = potential_instance_ids
        
        self.Annotations = []

        for ind, obj_id in enumerate(instance_ids_to_process):
            positions = (all_instance_msk == obj_id) #480,640 array 0-1位置
            if not np.any(positions): # Skip if this object_id results in an empty mask (should not happen if obj_id came from unique_values)
                continue

            expanded_position = np.expand_dims(positions,axis=0) #1,480,640 0-1位置
            expanded_position = np.repeat(expanded_position, grasp.shape[0], axis=0) #3,480,640 0-1位置
            
            filted_grasp = np.where(expanded_position, grasp, 0)  #3,480,640
            
            try:
                point = self.sample_point(positions)
                bbox = self.sample_box(positions)
            except ValueError: # Handle cases where mask might be empty after all, or sample_point/box fails
                print(f"Warning: Could not sample point/bbox for obj_id {obj_id} in {ins_mask_path}. Skipping this instance.")
                continue

            seg = GraspMat.decode(filted_grasp) # Output is (4, H, W)

            #add sematic mask as the 5th channel
            semantic_mask_channel = np.expand_dims(positions.astype(np.float32), axis=0) # (1, H, W)
            seg = np.concatenate((seg, semantic_mask_channel), axis=0) # Resulting shape (5, H, W)
            
            segmentation_dict = {
                'size' : [480,640], # Height, Width
                'counts' : seg # Now (5, H, W)
            }

            ind_dict = {
                'bbox' : [x for point_coords in bbox for x in point_coords],   # Flattened [min_row, min_col, max_row, max_col]
                'area' : float(np.sum(positions)), # Area of the instance mask
                'segmentation' : segmentation_dict,
                'predicted_iou' : None, # Placeholder
                'point_coords' : [list(point)],    # [[row, col]]
                'crop_box' : None, # Placeholder
                'id' : int(obj_id), # Store the instance ID
                'stability_score' : None, # Placeholder
            }

            self.Annotations.append(ind_dict)


    def annotations(self):
        return self.Annotations


    @staticmethod
    def sample_point(positions)-> tuple:
        """
            从mask为True的区域中随机采样一个点，并返回其 (row, col) 坐标。

            Args:
                mask (np.ndarray): 布尔型 mask，形状为 (H, W)

            Returns:
                tuple: 随机采样的点坐标 (row, col)
            """
        # 找到 mask 为 True 的所有坐标
        coords = np.argwhere(positions)
        if coords.size == 0:
            raise ValueError("Mask中没有True的区域 (sample_point)")
        # 随机选择一个索引
        idx = np.random.choice(len(coords))
        point = tuple(coords[idx]) # (row, col)
        return point

    @staticmethod
    def sample_box(positions)-> tuple:
        """
        计算mask的最小包围框，并返回包围框的左上和右下角点坐标。
        Args:
            mask (np.ndarray): 布尔型 mask，形状为 (H, W)
        Returns:
            tuple: ((min_row, min_col), (max_row, max_col))
        """
        coords = np.argwhere(positions)
        if coords.size == 0:
            raise ValueError("Mask中没有True的区域 (sample_box)")

        # 分别取出所有行和列索引
        rows = coords[:, 0]
        cols = coords[:, 1]
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())

        return ((min_row, min_col), (max_row, max_col))


    @staticmethod
    def visualize_mask_bbox_point(mask: np.ndarray, bbox: tuple, point: tuple):
        import matplotlib.pyplot as plt
        """
        可视化 mask，并在图上绘制出包围框和采样点。

        Args:
            mask: 二值 mask，形状 (H, W)，dtype 为 bool
            bbox: 包围框，格式 ((min_row, min_col), (max_row, max_col))
            point: 采样到的点坐标 (row, col)
        """
        plt.figure(figsize=(8, 6))
        # 显示 mask，使用灰度图
        plt.imshow(mask, cmap='gray')

        # 绘制包围框（注意：imshow坐标系中，x对应列，y对应行）
        (min_row, min_col), (max_row, max_col) = bbox
        width = max_col - min_col
        height = max_row - min_row
        rect = plt.Rectangle((min_col, min_row), width, height,
                             edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)

        # 绘制采样点
        # 注意scatter的x对应列，y对应行
        plt.scatter(point[1], point[0], color='blue', s=100)

        plt.title("Mask with Bounding Box and Sampled Point")
        plt.axis("off")
        plt.show()


    @staticmethod
    def _decode(mat):
        """
        解析 grasp_mat (pos, angle, width)
        Args:
            mat: np.ndarray (3, h, w)
        Returns:
                (4, h, w)  float: (confidence, cos(2*angle), sin(2*angle), width_normalized)
        """
        if mat.shape[0] != 3:
            raise ValueError(f"_decode expects 3 channels (pos, angle, width), got {mat.shape[0]}")

        h, w = mat.shape[1:]
        grasp_confidence = mat[0, :, :]
        grasp_angle_rad = mat[1, :, :] # Should be in radians
        grasp_width_px = mat[2, :, :]

        # Create cos and sin maps for the angle
        # angle_mat will store cos(2*angle) and sin(2*angle)
        angle_representation = np.zeros((2, h, w), dtype=np.float64) 
        
        # Apply cos(2*angle) and sin(2*angle) where confidence is > 0
        # For areas with no grasp (confidence == 0), cos and sin will remain 0.
        valid_grasp_points = grasp_confidence > 0
        angle_representation[0, valid_grasp_points] = np.cos(2 * grasp_angle_rad[valid_grasp_points])
        angle_representation[1, valid_grasp_points] = np.sin(2 * grasp_angle_rad[valid_grasp_points])

        # Prepare the 4-channel output matrix
        # Channel 0: grasp_confidence
        # Channel 1: cos(2*angle)
        # Channel 2: sin(2*angle)
        # Channel 3: normalized_width (e.g., by image width or a fixed factor like 100.0 or 150.0 as in some datasets)
        # Using 100.0 as per original code's ret_mat[-1, :, :] = grasp_width / 100.
        # Ensure width is also zeroed out where confidence is zero.
        normalized_width = np.zeros_like(grasp_width_px)
        normalized_width[valid_grasp_points] = grasp_width_px[valid_grasp_points] / 100.0

        ret_mat = np.stack([
            grasp_confidence,
            angle_representation[0],
            angle_representation[1],
            normalized_width
        ], axis=0)

        return ret_mat

    @staticmethod
    def decode(grasp_mat_3_channel):
        """
        Decodes a 3-channel grasp matrix (pos, angle, width) into a 4-channel representation.
        Args:
            grasp_mat_3_channel (np.ndarray): (3, H, W)
        Returns:
            np.ndarray: (4, H, W) -> (confidence, cos(2*angle), sin(2*angle), normalized_width)
        """
        return GraspMat._decode(grasp_mat_3_channel)
