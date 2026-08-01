from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
from logging import Logger
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from .config import WorkflowConfig


class BaseWorkflow(ABC):
    """A base class for managing Vision AI workflow.

    Attributes:
        LineString: Class for creating line string geometries from shapely.
        Polygon: Class for creating polygon geometries from shapely.
        Point: Class for creating point geometries from shapely.
        prep: Prepared geometry function from shapely for optimized spatial operations.
        cfg (dict[str, Any]): Configuration dictionary loaded from YAML file and updated with kwargs.
        LOGGER: Logger instance for solution-specific logging.
        annotator: Annotator instance for drawing on images.
        tracks: YOLO tracking results from the latest inference.
        track_data: Extracted tracking data (boxes or OBB) from tracks.
        boxes (list): Bounding box coordinates from tracking results.
        clss (list[int]): Class indices from tracking results.
        track_ids (list[int]): Track IDs from tracking results.
        confs (list[float]): Confidence scores from tracking results.
        track_line: Current track line for storing tracking history.
        masks: Segmentation masks from tracking results.
        r_s: Region or line geometry object for spatial operations.
        frame_no (int): Current frame number for logging purposes.
        region (list[tuple[int, int]]): List of coordinate tuples defining region of interest.
        line_width (int): Width of lines used in visualizations.
        model (YOLO): Loaded YOLO model instance.
        names (dict[int, str]): Dictionary mapping class indices to class names.
        classes (list[int]): List of class indices to track.
        show_conf (bool): Flag to show confidence scores in annotations.
        show_labels (bool): Flag to show class labels in annotations.
        device (str): Device for model inference.
        track_add_args (dict[str, Any]): Additional arguments for tracking configuration.
        env_check (bool): Flag indicating whether environment supports image display.
        track_history (defaultdict): Dictionary storing tracking history for each object.
        profilers (tuple): Profiler instances for performance monitoring.

    Methods:
        adjust_box_label: Generate formatted label for bounding box.
        extract_tracks: Apply object tracking and extract tracks from input image.
        store_tracking_history: Store object tracking history for given track ID and bounding box.
        initialize_region: Initialize counting region and line segment based on configuration.
        display_output: Display processing results including frames or saved results.
        process: Process method to be implemented by each Solution subclass.
    """
    def __init__(self, cfg: WorkflowConfig):
        self.cfg = cfg
        self.logger: Logger

        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # Initialize annotator
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1  # Only for logging

        self.region = self.cfg.region  # Store region data for other classes usage
        self.line_width = self.cfg.line_width

        # Load Model and store additional information (classes, show_conf, show_label)
        if self.cfg.model is None:
            self.cfg.model = "yolo26n.pt"
        self.model = None
        self.names = self.model.names
        self.classes = self.cfg.classes
        self.show_conf = self.cfg.show_conf
        self.show_labels = self.cfg.show_labels
        self.device = self.cfg.device

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: getattr(self.cfg, k) for k in {"iou", "conf", "device", "max_det", "quantize", "tracker", "imgsz"}
        }  # verbose must be passed to track method; setting it False in YOLO still logs the track information.

        # Initialize environment and region setup
        self.env_check: bool = False
        self.track_history = defaultdict(list)
        self.profilers = []

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        """Generate a formatted label for a bounding box.

        This method constructs a label string for a bounding box using the class index and confidence score. Optionally
        includes the track ID if provided. The label format adapts based on the display settings defined in
        `self.show_conf` and `self.show_labels`.

        Args:
            cls (int): The class index of the detected object.
            conf (float): The confidence score of the detection.
            track_id (int, optional): The unique identifier for the tracked object.

        Returns:
            (str | None): The formatted label string if `self.show_labels` is True; otherwise, None.
        """
        name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None

    def extract_tracks(self, im0: np.ndarray) -> None:
        """Apply object tracking and extract tracks from an input image or frame.

        Args:
            im0 (np.ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        is_obb = self.tracks.obb is not None
        self.track_data = self.tracks.obb if is_obb else self.tracks.boxes  # Extract tracks for OBB or object detection

        if self.track_data and self.track_data.is_track:
            self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.logger.warning("No tracks found.")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    def store_tracking_history(self, track_id: int, box) -> None:
        """Store the tracking history of an object.

        This method updates the tracking history for a given object by appending the center point of its bounding box to
        the track line. It maintains a maximum of 30 points in the tracking history.

        Args:
            track_id (int): The unique identifier for the tracked object.
            box (list[float]): The bounding box coordinates of the object in the format [x1, y1, x2, y2].

        Examples:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]
        self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    @staticmethod
    def get_enclosing_box(box: torch.Tensor | list[float]) -> torch.Tensor | list[float]:
        """Return the axis-aligned box [x1, y1, x2, y2] enclosing a box extracted by `extract_tracks`.

        Boxes from OBB models are (4, 2) xyxyxyxy corner points, while boxes from detection models are already
        axis-aligned [x1, y1, x2, y2]. This method normalizes both formats to [x1, y1, x2, y2] for solutions that
        require axis-aligned coordinates, e.g. for image slicing or box centers.

        Args:
            box (torch.Tensor | list[float]): Bounding box in [x1, y1, x2, y2] format or (4, 2) OBB corner points.

        Returns:
            (torch.Tensor | list[float]): Axis-aligned bounding box in [x1, y1, x2, y2] format.

        Examples:
            >>> import torch
            >>> BaseSolution.get_enclosing_box(torch.tensor([[2.0, 1.0], [4.0, 3.0], [2.0, 5.0], [0.0, 3.0]]))
            tensor([0., 1., 4., 5.])
        """
        return torch.cat([box.amin(0), box.amax(0)]) if isinstance(box, torch.Tensor) and box.numel() > 4 else box

    def initialize_region(self) -> None:
        """Initialize the counting region and line segment based on configuration settings."""
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # region or line

    def display_output(self, plot_im: np.ndarray) -> None:
        """Display the results of the processing, which could involve showing frames, printing counts, or saving
        results.

        This method is responsible for visualizing the output of the object detection and tracking process. It displays
        the processed frame with annotations, and allows for user interaction to close the display.

        Args:
            plot_im (np.ndarray): The image or frame that has been processed and annotated.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        Notes:
            - This method will only display output if the 'show' configuration is set to True and the environment
              supports image display.
            - The display can be closed by pressing the 'q' key.
        """
        if self.cfg.show and self.env_check:
            cv2.imshow("Ultralytics Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # Closes current frame window
                return

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any):
        """Process method should be implemented by each Solution subclass."""
        pass

    def __call__(self, *args: Any, **kwargs: Any):
        """Allow instances to be called like a function with flexible arguments."""
        with self.profilers[1]:
            result = self.process(*args, **kwargs)  # Call the subclass-specific process method
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # solution time = process - track
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)  # Only for logging.
            # Use model input shape (reflects imgsz) if predictor is available
            if hasattr(self.model, "predictor") and self.model.predictor and hasattr(self.model.predictor, "imgsz"):
                input_h, input_w = self.model.predictor.imgsz
            else:
                input_h, input_w = result.plot_im.shape[:2]
            self.logger.info(
                f"{self.frame_no}: {input_h}x{input_w} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape "
                f"(1, {getattr(self.model, 'channels', 3)}, {input_h}, {input_w})\n"
            )
        return result


class WorkflowResults:
    """A class to encapsulate the results of Ultralytics Solutions.

    This class is designed to store and manage various outputs generated by the solution pipeline, including counts,
    angles, workout stages, and other analytics data. It provides a structured way to access and manipulate results from
    different computer vision solutions such as object counting, pose estimation, and tracking analytics.

    Attributes:
        plot_im (np.ndarray): Processed image with counts, blurred, or other effects from solutions.
        in_count (int): The total number of "in" counts in a video stream.
        out_count (int): The total number of "out" counts in a video stream.
        classwise_count (dict[str, int]): A dictionary containing counts of objects categorized by class.
        queue_count (int): The count of objects in a queue or waiting area.
        workout_count (list[int]): Per-track workout repetition counts (one entry per currently tracked individual).
        workout_angle (list[float]): Per-track exercise angles for currently tracked individuals.
        workout_stage (list[str]): Per-track current exercise stage for currently tracked individuals.
        pixels_distance (float): The calculated distance in pixels between two points or objects.
        available_slots (int): The number of available slots in a monitored area.
        filled_slots (int): The number of filled slots in a monitored area.
        email_sent (bool): A flag indicating whether an email notification was sent.
        total_tracks (int): The total number of tracked objects.
        region_counts (dict[str, int]): The count of objects within a specific region.
        speed_dict (dict[str, float]): A dictionary containing speed information for tracked objects.
        total_crop_objects (int): Total number of cropped objects using ObjectCropper class.
        speed (dict[str, float]): Performance timing information for tracking and solution processing.
    """

    def __init__(self, **kwargs):
        """Initialize a SolutionResults object with default or user-specified values.

        Args:
            **kwargs (Any): Optional arguments to override default attribute values.
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = []
        self.workout_angle = []
        self.workout_stage = []
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}  # for speed estimation
        self.total_crop_objects = 0
        self.speed = {}

        # Override with user-defined values
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """Return a formatted string representation of the SolutionResults object.

        Returns:
            (str): A string representation listing non-null attributes.
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # Exclude `plot_im` explicitly
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
