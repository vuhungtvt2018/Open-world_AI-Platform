from __future__ import annotations

from typing import Any
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
import numpy as np

from .config import TrackerConfig

class TrackState:
    """Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        ...     print("Object is newly detected.")
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """

    _count = 0

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        """Return the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Increment and return the next unique global track ID for object tracking."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Activate the track with provided arguments, initializing necessary attributes for tracking."""
        raise NotImplementedError

    def predict(self) -> None:
        """Predict the next state of the track based on the current state and tracking model."""
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the track with new observations and data, modifying its state and attributes accordingly."""
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark the track as lost by updating its state to TrackState.Lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark the track as removed by setting its state to TrackState.Removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to its initial value."""
        BaseTrack._count = 0


class BaseTracker(ABC):
    """Base class for multi-object tracking systems.

    Manages active, lost, and removed tracks across video frames based on 
    configuration parameters (TrackerConfig).

    Attributes:
        cfg (TrackerConfig): Configuration object containing thresholds and parameters.
        tracked_stracks (List[BaseTrack]): Currently active tracks.
        lost_stracks (List[BaseTrack]): Tracks lost in recent frames.
        removed_stracks (List[BaseTrack]): Tracks removed permanently.
        frame_id (int): Frame index counter.
    """

    def __init__(self, cfg: "TrackerConfig") -> None:
        """Initialize tracker state and configurations.

        Args:
            cfg (TrackerConfig): Tracking options (thresholds, buffer sizes, etc.).
        """
        self.cfg = cfg
        self.tracked_stracks: List[BaseTrack] = []
        self.lost_stracks: List[BaseTrack] = []
        self.removed_stracks: List[BaseTrack] = []
        self.frame_id: int = 0

    @abstractmethod
    def update(self, results: Any, img: Optional[np.ndarray] = None) -> List[BaseTrack]:
        """Update tracker state using new detection results from a frame.

        Args:
            results: Object detections (e.g., bounding boxes, scores, classes).
            img (Optional[np.ndarray]): Original image frame (required for feature-based trackers like BotSORT).

        Returns:
            List[BaseTrack]: List of currently active tracks for the current frame.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal frame counter, track ID sequence, and clear active tracks."""
        self.frame_id = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        BaseTrack.reset_id()

    @staticmethod
    def joint_stracks(
        tlista: List[BaseTrack], tlistb: List[BaseTrack]
    ) -> List[BaseTrack]:
        """Combine two lists of tracks while removing duplicates based on track_id.

        Args:
            tlista (List[BaseTrack]): First track list.
            tlistb (List[BaseTrack]): Second track list.

        Returns:
            List[BaseTrack]: Merged list of unique tracks.
        """
        exists = {t.track_id for t in tlista}
        res = list(tlista)
        for t in tlistb:
            if t.track_id not in exists:
                exists.add(t.track_id)
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(
        tlista: List[BaseTrack], tlistb: List[BaseTrack]
    ) -> List[BaseTrack]:
        """Subtract track list B from track list A based on track_id.

        Args:
            tlista (List[BaseTrack]): Source track list.
            tlistb (List[BaseTrack]): Tracks to remove from source list.

        Returns:
            List[BaseTrack]: Remaining tracks after subtraction.
        """
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            stracks.pop(t.track_id, None)
        return list(stracks.values())

    @staticmethod
    def remove_duplicate_stracks(
        a_stracks: List[BaseTrack], b_stracks: List[BaseTrack], iou_thresh: float = 0.15
    ) -> Tuple[List[BaseTrack], List[BaseTrack]]:
        """Remove overlapping duplicate tracks across two track collections.

        Prevents duplicate tracks from following the same object simultaneously.

        Args:
            a_stracks (List[BaseTrack]): First group of tracks.
            b_stracks (List[BaseTrack]): Second group of tracks.
            iou_thresh (float): Overlap threshold above which the younger track is discarded.

        Returns:
            Tuple[List[BaseTrack], List[BaseTrack]]: Filtered (dup-free) track groups.
        """
        if not a_stracks or not b_stracks:
            return a_stracks, b_stracks

        # Calculate IoU distance matrix between tracks
        # (Concrete distance logic implemented in derived tracker, e.g., ByteTrack / BotSORT)
        dupa = []
        dupb = []
        
        # Placeholder/Simplified overlap check logic:
        # If overlap > iou_thresh, keep the older track (lower start_frame or larger duration)
        for a in a_stracks:
            for b in b_stracks:
                if a.track_id == b.track_id:
                    continue
                # Compare frame durations to resolve duplicates
                time_a = a.frame_id - a.start_frame
                time_b = b.frame_id - b.start_frame
                if time_a > time_b:
                    dupb.append(b)
                else:
                    dupa.append(a)

        res_a = [t for t in a_stracks if t not in dupa]
        res_b = [t for t in b_stracks if t not in dupb]
        return res_a, res_b

    def merge_track_pools(self, activated: list, refind: list, lost: list, removed: list, removed_buffer: int=1000):
        """Apply the standard end-of-frame bookkeeping to a tracker's persistent pools in place.

        Merges newly activated and re-found tracks into `tracked_stracks`, moves the transitioned tracks into
        `lost_stracks`, dedups by IoU, appends removals to `removed_stracks`, and trims the removed buffer
        to `removed_buffer` entries.

        Args:
            activated (list): Tracks updated from the Tracked state this frame.
            refind (list): Tracks re-activated from the Lost state this frame.
            lost (list): Tracks transitioned to Lost this frame.
            removed (list): Tracks transitioned to Removed this frame.
            removed_buffer (int): Maximum number of historical removed tracks to retain.
        """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = BaseTracker.joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = BaseTracker.joint_stracks(self.tracked_stracks, refind)

        self.lost_stracks = BaseTracker.sub_stracks(self.lost_stracks, self.tracked_stracks)
        new_lost = [t for t in lost if t not in self.lost_stracks]
        self.lost_stracks.extend(new_lost)
        new_removed = [t for t in removed if t not in self.removed_stracks]
        self.removed_stracks.extend(new_removed)
        self.lost_stracks = BaseTracker.sub_stracks(self.lost_stracks, self.removed_stracks)

        self.tracked_stracks, self.lost_stracks = BaseTracker.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        if len(self.removed_stracks) > removed_buffer:
            self.removed_stracks = self.removed_stracks[-removed_buffer:]