# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Optional, List, Tuple

import numpy as np

from vision_ai_platform.packages.utils import LOGGER
from vision_ai_platform.packages.utils.ops import xywh2ltwh
from vision_ai_platform.packages.core.tracker import BaseTrack, TrackState, BaseTracker
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes


class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data and update its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0], score=0.9, cls=0)
            >>> new_track = STrack([105, 205, 55, 85, 0], score=0.95, cls=0)
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh  # already a fresh array, safe to mutate
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker(BaseTracker):
    """BYTETracker: A tracking algorithm built on top of YOLO for object detection and tracking.

    Inherits core tracking state structures and utility operations from BaseTracker.
    """

    track_class = STrack

    def __init__(self, cfg: Any) -> None:
        """Initialize a BYTETracker instance for object tracking.

        Args:
            cfg (TrackerConfig): Configuration containing tracking parameters.
        """
        super().__init__(cfg)
        self.cfg = cfg  # Retain self.cfg reference if required by internal methods
        self.max_frames_lost = getattr(cfg, "track_buffer", 30)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(
        self,
        results: Any,
        img: Optional[np.ndarray] = None,
        feats: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[STrack]:
        """Update tracker state with new detections for a single frame.

        Returns:
            List[STrack]: List of currently active tracks for the current frame.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        results_high, results_low, mask_high, mask_low = self._split_detections(results)
        detections = self.init_track(results_high, self._input_for(img, feats, mask_high))
        detections_second = self.init_track(results_low, self._input_for(img, feats, mask_low))

        for tracks, mask in ((detections, mask_high), (detections_second, mask_low)):
            for track, i in zip(tracks, np.flatnonzero(mask)):
                track.idx = i  # preserve detection index mapping

        unconfirmed, tracked_stracks = self._split_tracked()
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        self._pre_first_associate(strack_pool, unconfirmed, img, results_high)

        u_track, u_detection = self._first_association(
            strack_pool, detections, activated_stracks, refind_stracks
        )
        u_track, u_detection = self._post_first_association(
            strack_pool, detections, u_track, u_detection, activated_stracks, refind_stracks
        )
        self._second_association(
            strack_pool, u_track, detections_second, activated_stracks, refind_stracks, lost_stracks
        )
        u_detection, detections = self._unconfirmed_association(
            unconfirmed, u_detection, detections, activated_stracks, removed_stracks
        )
        self._init_new_tracks(u_detection, detections, activated_stracks, refind_stracks)
        self._remove_stale_lost(removed_stracks)

        # Apply end-of-frame track pool updates using BaseTracker method
        self.merge_track_pools(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

        return self._format_output()

    def _split_detections(self, results: Any) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        """Split detections into high-confidence and low-confidence subsets."""
        scores = results.conf
        high_thresh = getattr(self.cfg, "track_high_thresh", 0.5)
        low_thresh = getattr(self.cfg, "track_low_thresh", 0.1)

        remain_inds = scores >= high_thresh
        inds_low = scores > low_thresh
        inds_below_high = scores < high_thresh
        return results[remain_inds], results[inds_low & inds_below_high], remain_inds, inds_low & inds_below_high

    def _input_for(self, img: Optional[np.ndarray], feats: Optional[np.ndarray], mask: np.ndarray) -> Any:
        """Return the per-detection auxiliary input for `init_track`."""
        if feats is not None and len(feats):
            return feats[mask]
        if getattr(self, "encoder", None) is not None and getattr(self.cfg, "model", "auto") == "auto":
            return None
        return img

    def _split_tracked(self) -> Tuple[List[STrack], List[STrack]]:
        """Separate `self.tracked_stracks` into unconfirmed and confirmed lists."""
        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)
        return unconfirmed, tracked

    def _pre_first_associate(
        self, strack_pool: List[STrack], unconfirmed: List[STrack], img: Optional[np.ndarray], results_high: Any
    ) -> None:
        """Hook executed after Kalman predict and before first-stage assignment."""
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results_high.xyxy)
            except Exception as e:
                LOGGER.warning(f"GMC failed, falling back to identity: {e}")
                warp = np.eye(2, 3)
            multi_gmc(strack_pool, warp)
            multi_gmc(unconfirmed, warp)

    def _first_association(
        self, strack_pool: List[STrack], detections: List[STrack], activated: List[STrack], refind: List[STrack]
    ) -> Tuple[List[int], List[int]]:
        """First-stage association between active pool and high-score detections."""
        dists = self.get_dists(strack_pool, detections)
        match_thresh = getattr(self.cfg, "match_thresh", 0.8)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh)
        self._apply_matches(matches, strack_pool, detections, activated, refind)
        return u_track, u_detection

    def _post_first_association(
        self,
        strack_pool: List[STrack],
        detections: List[STrack],
        u_track: List[int],
        u_detection: List[int],
        activated: List[STrack],
        refind: List[STrack],
    ) -> Tuple[List[int], List[int]]:
        """Hook executed after first stage and before second stage association."""
        return u_track, u_detection

    def _apply_matches(
        self,
        matches: Any,
        pool: List[STrack],
        detections: List[STrack],
        activated: List[STrack],
        refind: List[STrack],
    ) -> None:
        """Apply matched track-detection pairs."""
        for itracked, idet in matches:
            self._apply_match(pool[itracked], detections[idet], activated, refind)

    def _apply_match(self, track: STrack, det: STrack, activated: List[STrack], refind: List[STrack]) -> None:
        """Update or re-activate a track with its matched detection."""
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)

    def _second_association(
        self,
        strack_pool: List[STrack],
        u_track: List[int],
        detections_second: List[STrack],
        activated: List[STrack],
        refind: List[STrack],
        lost: List[STrack],
    ) -> None:
        """Second-stage association with low-score detections using IoU distance."""
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked_stracks and detections_second:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked_stracks)))

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

    def _unconfirmed_association(
        self,
        unconfirmed: List[STrack],
        u_detection: List[int],
        detections: List[STrack],
        activated: List[STrack],
        removed: List[STrack],
    ) -> Tuple[List[int], List[STrack]]:
        """Associate unconfirmed tracks with remaining high-score detections."""
        detections = [detections[i] for i in u_detection]
        if not unconfirmed:
            return list(range(len(detections))), detections
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)
        return u_detection, detections

    def _init_new_tracks(
        self,
        u_detection: List[int],
        detections: List[STrack],
        activated: List[STrack],
        refind: Optional[List[STrack]] = None,
    ) -> None:
        """Activate new tracks from unmatched high-score detections."""
        new_thresh = getattr(self.cfg, "new_track_thresh", 0.6)
        for inew in u_detection:
            track = detections[inew]
            if track.score < new_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

    def _remove_stale_lost(self, removed: List[STrack]) -> None:
        """Remove tracks in lost state beyond `max_frames_lost` threshold."""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed.append(track)

    def _format_output(self) -> List[STrack]:
        """Format and return current active track objects matching standard BaseTracker interface."""
        return [x for x in self.tracked_stracks if x.is_activated]

    def get_kalmanfilter(self) -> Any:
        """Return initialized Kalman filter object."""
        return KalmanFilterXYAH()

    def init_track(self, results: Any, img: Optional[np.ndarray] = None) -> List[STrack]:
        """Initialize STrack instances from standard detection results."""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [self.track_class(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        """Calculate IoU-based distance matrix with optional score fusion."""
        dists = matching.iou_distance(tracks, detections)
        if getattr(self.cfg, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: List[STrack]) -> None:
        """Perform multi-track Kalman prediction."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id() -> None:
        """Reset the ID counter for STrack instances."""
        STrack.reset_id()

    def reset(self) -> None:
        """Reset internal frame counter, track lists, and re-initialize state."""
        super().reset()
        self.kalman_filter = self.get_kalmanfilter()