#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GraspSTrack and GraspBYTETracker

Extended ByteTrack to support grasp parameter tracking.
Each tracked object carries 6-DOF grasp parameters that are updated with detections.

Grasp parameters: [grasp_dx, grasp_dy, cos2theta, sin2theta, width, score]
    grasp_dx, grasp_dy: normalized offset from bbox center to grasp center
        grasp_center_x = bbox_cx + grasp_dx * bbox_w
        grasp_center_y = bbox_cy + grasp_dy * bbox_h
"""

import numpy as np
from collections import deque
import torch

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState


class GraspSTrack(BaseTrack):
    """
    Extended STrack with 6-DOF grasp parameters.

    Attributes:
        grasp_params: dict with keys ['grasp_dx', 'grasp_dy', 'cos2theta', 'sin2theta', 'width', 'score']
        grasp_history: deque of historical grasp params for temporal smoothing
    """

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, grasp_params=None):
        """
        Args:
            tlwh: Bounding box [top_left_x, top_left_y, width, height]
            score: Detection confidence
            grasp_params: dict with 6-DOF grasp parameters, or None
        """
        # Original STrack attributes
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

        # Grasp parameters (6-DOF)
        if grasp_params is None:
            self.grasp_params = {
                'grasp_dx': 0.0,
                'grasp_dy': 0.0,
                'cos2theta': 1.0,
                'sin2theta': 0.0,
                'width': 0.3,
                'score': 0.0
            }
        else:
            self.grasp_params = grasp_params.copy()

        # History for temporal smoothing
        self.grasp_history = deque(maxlen=10)
        if grasp_params is not None:
            self.grasp_history.append(grasp_params.copy())

    @property
    def grasp_center(self):
        """Compute grasp center from bbox and offset."""
        bbox_cx = self._tlwh[0] + self._tlwh[2] / 2
        bbox_cy = self._tlwh[1] + self._tlwh[3] / 2
        grasp_x = bbox_cx + self.grasp_params['grasp_dx'] * self._tlwh[2]
        grasp_y = bbox_cy + self.grasp_params['grasp_dy'] * self._tlwh[3]
        return grasp_x, grasp_y

    @property
    def grasp_angle(self):
        """Decode grasp angle from cos/sin encoding (radians)."""
        cos2theta = self.grasp_params['cos2theta']
        sin2theta = self.grasp_params['sin2theta']
        return 0.5 * np.arctan2(sin2theta, cos2theta)

    @property
    def grasp_angle_deg(self):
        """Grasp angle in degrees."""
        return np.degrees(self.grasp_angle)

    @property
    def grasp_width(self):
        """Grasp width in pixels (denormalized)."""
        return self.grasp_params['width'] * 100.0

    @property
    def grasp_score(self):
        """Grasp confidence score."""
        return self.grasp_params['score']

    def get_grasp_pose(self):
        """
        Get full grasp pose.

        Returns:
            dict with 'x', 'y', 'angle', 'width', 'score'
            where (x, y) is the actual grasp center
        """
        grasp_x, grasp_y = self.grasp_center

        return {
            'x': float(grasp_x),
            'y': float(grasp_y),
            'grasp_dx': float(self.grasp_params['grasp_dx']),
            'grasp_dy': float(self.grasp_params['grasp_dy']),
            'angle': float(self.grasp_angle),
            'angle_deg': float(self.grasp_angle_deg),
            'width': float(self.grasp_width),
            'score': float(self.grasp_score),
        }

    def get_smoothed_grasp(self, alpha=0.7):
        """
        Get temporally smoothed grasp parameters using exponential moving average.

        Args:
            alpha: Decay factor (higher = more weight on recent values)

        Returns:
            dict with smoothed grasp parameters
        """
        if len(self.grasp_history) == 0:
            return self.grasp_params.copy()

        smoothed = {
            'grasp_dx': 0.0,
            'grasp_dy': 0.0,
            'cos2theta': 0.0,
            'sin2theta': 0.0,
            'width': 0.0,
            'score': 0.0
        }

        weight = 1.0
        total_weight = 0.0

        for grasp in reversed(list(self.grasp_history)):
            for key in smoothed:
                smoothed[key] += weight * grasp[key]
            total_weight += weight
            weight *= alpha

        for key in smoothed:
            smoothed[key] /= total_weight

        return smoothed

    def predict(self):
        """Kalman filter prediction step."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        """Batch Kalman prediction for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = GraspSTrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track with new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()

        self.score = new_track.score
        self._update_grasp_params(new_track.grasp_params)

    def update(self, new_track, frame_id):
        """Update a matched track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        self._update_grasp_params(new_track.grasp_params)

    def _update_grasp_params(self, new_grasp_params):
        """Update grasp parameters and add to history."""
        if new_grasp_params is not None:
            self.grasp_params = new_grasp_params.copy()
            self.grasp_history.append(new_grasp_params.copy())

    @property
    def tlwh(self):
        """Get current bbox in (top left x, top left y, width, height) format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Get current bbox in (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to (center x, center y, aspect ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'GraspTrack_{self.track_id}_({self.start_frame}-{self.end_frame})'

    def to_dict(self):
        """Export track as dictionary."""
        return {
            'track_id': self.track_id,
            'bbox': self.tlbr.tolist(),
            'tlwh': self.tlwh.tolist(),
            'score': float(self.score),
            'grasp': self.get_grasp_pose(),
            'state': self.state.name if hasattr(self.state, 'name') else str(self.state),
        }


class GraspBYTETracker:
    """
    Extended BYTETracker with grasp parameter support.

    Tracks objects with associated 6-DOF grasp parameters.
    """

    def __init__(self, args, frame_rate=30):
        """
        Args:
            args: Namespace with tracking parameters
                - track_thresh: Detection confidence threshold
                - track_buffer: Number of frames to keep lost tracks
                - match_thresh: IoU threshold for matching
                - mot20: Whether to use MOT20 settings
            frame_rate: Video frame rate
        """
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks (can be recovered)
        self.removed_stracks = []  # Removed tracks

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        """
        Update tracker with new detections.

        Args:
            output_results: Detection results, either:
                - Tensor/array of shape (N, 5+num_cls+6):
                  [x1, y1, x2, y2, obj_conf, cls_conf..., grasp_dx, grasp_dy, cos2theta, sin2theta, width, score]
                - Tensor/array of shape (N, 6+6) for single class:
                  [x1, y1, x2, y2, obj_conf, cls_conf, grasp_dx, grasp_dy, cos2theta, sin2theta, width, score]
            img_info: Tuple of (original_height, original_width)
            img_size: Network input size (height, width)

        Returns:
            List[GraspSTrack]: Active tracks with grasp parameters
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Convert to numpy if tensor
        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()

        if len(output_results) == 0:
            # No detections - just update states
            for track in self.tracked_stracks:
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # Parse detection results
            # Assume format: [x1, y1, x2, y2, obj*cls, cls_id, grasp_dx, grasp_dy, cos2t, sin2t, width, score]
            # or [x1, y1, x2, y2, obj, cls, grasp_dx, grasp_dy, cos2t, sin2t, width, score]

            if output_results.shape[1] >= 12:
                # Format with 6-DOF grasp: [bbox(4), obj(1), cls(1), grasp(6)]
                bboxes = output_results[:, :4]
                scores = output_results[:, 4] * output_results[:, 5]
                grasp_data = output_results[:, -6:]  # Last 6 columns
            elif output_results.shape[1] >= 10:
                # Legacy format with 4-DOF grasp: [bbox(4), obj(1), cls(1), grasp(4)]
                bboxes = output_results[:, :4]
                scores = output_results[:, 4] * output_results[:, 5]
                grasp_data_4 = output_results[:, -4:]  # Last 4 columns
                # Pad to 6-DOF: add grasp_dx=0, grasp_dy=0
                grasp_data = np.zeros((len(output_results), 6))
                grasp_data[:, 2:6] = grasp_data_4
            elif output_results.shape[1] == 5:
                # Simple format: [x1, y1, x2, y2, score]
                bboxes = output_results[:, :4]
                scores = output_results[:, 4]
                grasp_data = np.zeros((len(output_results), 6))
            else:
                # Try to parse as [bbox(4), score, ...rest..., grasp(6)]
                bboxes = output_results[:, :4]
                scores = output_results[:, 4]
                if output_results.shape[1] > 10:
                    grasp_data = output_results[:, -6:]
                else:
                    grasp_data = np.zeros((len(output_results), 6))

            # Scale bboxes to original image size
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

            # Split into high and low confidence detections
            remain_inds = scores > self.args.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.args.track_thresh
            inds_second = np.logical_and(inds_low, inds_high)

            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            grasp_keep = grasp_data[remain_inds]

            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            grasp_second = grasp_data[inds_second]

            # Create detection objects (6-DOF grasp parameters)
            if len(dets) > 0:
                detections = [
                    GraspSTrack(
                        GraspSTrack.tlbr_to_tlwh(tlbr),
                        s,
                        {
                            'grasp_dx': float(g[0]),
                            'grasp_dy': float(g[1]),
                            'cos2theta': float(g[2]),
                            'sin2theta': float(g[3]),
                            'width': float(g[4]),
                            'score': float(g[5]) if len(g) > 5 else 0.0
                        }
                    )
                    for tlbr, s, g in zip(dets, scores_keep, grasp_keep)
                ]
            else:
                detections = []

            # Separate confirmed and unconfirmed tracks
            unconfirmed = []
            tracked_stracks = []
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

            # Step 1: First association with high score detections
            strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
            GraspSTrack.multi_predict(strack_pool)

            dists = matching.iou_distance(strack_pool, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)

            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.args.match_thresh
            )

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # Step 2: Second association with low score detections
            if len(dets_second) > 0:
                detections_second = [
                    GraspSTrack(
                        GraspSTrack.tlbr_to_tlwh(tlbr),
                        s,
                        {
                            'grasp_dx': float(g[0]),
                            'grasp_dy': float(g[1]),
                            'cos2theta': float(g[2]),
                            'sin2theta': float(g[3]),
                            'width': float(g[4]),
                            'score': float(g[5]) if len(g) > 5 else 0.0
                        }
                    )
                    for tlbr, s, g in zip(dets_second, scores_second, grasp_second)
                ]
            else:
                detections_second = []

            r_tracked_stracks = [
                strack_pool[i] for i in u_track
                if strack_pool[i].state == TrackState.Tracked
            ]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(
                dists, thresh=0.5
            )

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # Mark unmatched tracks as lost
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            # Step 3: Deal with unconfirmed tracks
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)

            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=0.7
            )

            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])

            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            # Step 4: Initialize new tracks
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

            # Step 5: Update state
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

        # Update track lists
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated
        ]

        return output_stracks

    def get_results(self):
        """Get all active tracks as list of dictionaries."""
        return [track.to_dict() for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista, tlistb):
    """Join two track lists, avoiding duplicates."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract tlistb from tlista."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks between two lists based on IoU."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]

    return resa, resb
