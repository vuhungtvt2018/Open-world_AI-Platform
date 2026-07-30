from vision_ai_platform.packages.core.tracker import TrackState, BaseTrack, BaseTracker
from vision_ai_platform.packages.core.config import TrackerConfig
from vision_ai_platform.packages.core.results import Results
from typing import Any, Optional, List, Tuple
import numpy as np
from vision_ai_platform.packages.utils.ops import linear_sum_assignment
from vision_ai_platform.packages.utils.loss import box_iou

class YOLOTracker(BaseTracker):
    def __init__(self, cfg: TrackerConfig):
        super().__init__(cfg)
        self.max_frames_lost = cfg.track_buffer

    def update(self, results: Results, img: Optional[np.ndarray] = None) -> List[BaseTrack]:
        """Update tracker state using new detection results from a frame.

        Args:
            results (Results): Object detections (e.g., bounding boxes, scores, classes).
            img (Optional[np.ndarray]): Original image frame (required for feature-based trackers like BotSORT).

        Returns:
            List[BaseTrack]: List of currently active tracks for the current frame.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Parse raw Ultralytics results format
        bboxes, scores, classes = self._parse_results(results)

        # 2. Split detections using Pydantic configuration thresholds
        high_indices = scores >= self.cfg.track_high_thresh
        low_indices = (scores >= self.cfg.track_low_thresh) & (
            scores < self.cfg.track_high_thresh
        )

        detections_high = self._init_tracks(
            bboxes[high_indices], scores[high_indices], classes[high_indices]
        )
        detections_low = self._init_tracks(
            bboxes[low_indices], scores[low_indices], classes[low_indices]
        )

        # 3. Separate tracked and lost track pools
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # 4. First stage association (High confidence detections vs All active/lost tracks)
        dists = self._iou_distance(strack_pool, detections_high)
        matches, u_track, u_det_high = self._linear_assignment(
            dists, thresh=self.cfg.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # 5. Second stage association (Unmatched tracked tracks vs Low confidence detections)
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = self._iou_distance(r_tracked_stracks, detections_low)
        matches, u_r_track, _ = self._linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for itracked in u_r_track:
            track = r_tracked_stracks[itracked]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 6. Associate unconfirmed tracks with remaining high-confidence detections
        dists = self._iou_distance(
            unconfirmed, [detections_high[i] for i in u_det_high]
        )
        matches, u_unconfirmed, u_detection = self._linear_assignment(
            dists, thresh=0.7
        )

        for itracked, idet in matches:
            unconfirmed[itracked].update(
                detections_high[u_det_high[idet]], self.frame_id
            )
            activated_stracks.append(unconfirmed[itracked])

        for itracked in u_unconfirmed:
            track = unconfirmed[itracked]
            track.mark_removed()
            removed_stracks.append(track)

        # 7. Initialize new tracks from unassociated high-confidence detections
        for idet in u_detection:
            track = detections_high[u_det_high[idet]]
            if track.score >= self.cfg.new_track_thresh:
                track.activate(self.frame_id)
                activated_stracks.append(track)

        # 8. Mark old lost tracks as removed (using track_buffer from Pydantic config)
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.cfg.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # 9. Apply persistent pool updates
        self.merge_track_pools(
            self,
            activated_stracks,
            refind_stracks,
            lost_stracks,
            removed_stracks,
            removed_buffer=1000,
        )

        return [t for t in self.tracked_stracks if t.is_activated]

    def _parse_results(
        self, results: Results
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bounding boxes, confidence scores, and class IDs from inputs."""
        if hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
        elif isinstance(results, np.ndarray) and results.shape[1] >= 6:
            boxes = results[:, :4]
            scores = results[:, 4]
            classes = results[:, 5]
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int32)
        return boxes, scores, classes

    def _init_tracks(
        self, bboxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
    ) -> List[BaseTrack]:
        """Instantiate BaseTrack objects in TLWH bounding box format."""
        tracks = []
        for bbox, score, cls in zip(bboxes, scores, classes):
            tlwh = np.array(
                [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                dtype=np.float32,
            )
            tracks.append(BaseTrack(tlwh, score, int(cls)))
        return tracks

    def _iou_distance(
        self, atracks: List[BaseTrack], btracks: List[BaseTrack]
    ) -> np.ndarray:
        """Compute IoU cost matrix."""
        if not atracks or not btracks:
            return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

        a_boxes = np.ascontiguousarray([t.tlbr for t in atracks], dtype=np.float32)
        b_boxes = np.ascontiguousarray([t.tlbr for t in btracks], dtype=np.float32)
        ious = box_iou(a_boxes, b_boxes)
        return 1.0 - ious

    def _linear_assignment(
        self, cost_matrix: np.ndarray, thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Solve bipartite matching via SciPy Hungarian Algorithm."""
        if cost_matrix.size == 0:
            return (
                [],
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])),
            )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_a = list(set(range(cost_matrix.shape[0])) - set(row_ind))
        unmatched_b = list(set(range(cost_matrix.shape[1])) - set(col_ind))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                unmatched_a.append(r)
                unmatched_b.append(c)
            else:
                matches.append((r, c))

        return matches, sorted(unmatched_a), sorted(unmatched_b)