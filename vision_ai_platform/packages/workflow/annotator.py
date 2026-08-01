from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from vision_ai_platform.packages.utils import LOGGER
from vision_ai_platform.packages.utils.annotator import Annotator


class WorkflowAnnotator(Annotator):
    """A specialized annotator class for visualizing and analyzing computer vision tasks.

    Attributes:
        im (np.ndarray): The image being annotated.
        line_width (int): Thickness of lines used in annotations.
        font_size (int): Size of the font used for text annotations.
        font (str): Path to the font file used for text rendering.
        pil (bool): Whether to use PIL for text rendering.
        example (str): Example text used to detect non-ASCII labels for PIL rendering.

    Methods:
        draw_region: Draw a region using specified points, colors, and thickness.
        queue_counts_display: Display queue counts in the specified region.
        display_analytics: Display overall statistics for parking lot management.
        estimate_pose_angle: Calculate the angle between three points in an object pose.
        draw_specific_kpts: Draw specific keypoints on the image.
        plot_workout_information: Draw a labeled text box on the image.
        plot_angle_and_count_and_stage: Visualize angle, step count, and stage for workout monitoring.
        plot_distance_and_line: Display the distance between centroids and connect them with a line.
        display_objects_labels: Annotate bounding boxes with object class labels.
        sweep_annotator: Visualize a vertical sweep line and optional label.
        visioneye: Map and connect object centroids to a visual "eye" point.
        adaptive_label: Draw a circular or rectangle background shape label in center of a bounding box.
    """
    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the WorkflowAnnotator class with an image for annotation.

        Args:
            im (np.ndarray): The image to be annotated.
            line_width (int, optional): Line thickness for drawing on the image.
            font_size (int, optional): Font size for text annotations.
            font (str): Path to the font file.
            pil (bool): Indicates whether to use PIL for rendering text.
            example (str): Example text used to detect non-ASCII labels for PIL rendering.
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(
        self,
        reg_pts: list[tuple[int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5,
    ):
        """Draw a region or line on the image.

        Args:
            reg_pts (list[tuple[int, int]], optional): Region points (for line 2 points, for region 4+ points).
            color (tuple[int, int, int]): BGR color value for the region (OpenCV format).
            thickness (int): Line thickness for drawing the region.
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def queue_counts_display(
        self,
        label: str,
        points: list[tuple[int, int]] | None = None,
        region_color: tuple[int, int, int] = (255, 255, 255),
        txt_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """Display queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): Queue counts label.
            points (list[tuple[int, int]], optional): Region points for center point calculation to display text.
            region_color (tuple[int, int, int]): BGR queue region color (OpenCV format).
            txt_color (tuple[int, int, int]): BGR text color (OpenCV format).
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
    ):
        """Display overall statistics for Solutions (e.g., parking management and object counting).

        Args:
            im0 (np.ndarray): Inference image.
            text (dict[str, Any]): Labels dictionary.
            txt_color (tuple[int, int, int]): Text color (BGR, OpenCV format).
            bg_color (tuple[int, int, int]): Background color (BGR, OpenCV format).
            margin (int): Gap between text and rectangle for better display.
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        """Convert a keypoint-like object to an (x, y) tuple of floats."""
        if hasattr(point, "detach"):  # torch.Tensor
            point = point.detach()
        if hasattr(point, "cpu"):  # torch.Tensor
            point = point.cpu()
        if hasattr(point, "numpy"):  # torch.Tensor
            point = point.numpy()
        if hasattr(point, "tolist"):  # numpy / torch
            point = point.tolist()
        return float(point[0]), float(point[1])

    @staticmethod
    @lru_cache(maxsize=256)
    def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """Calculate the angle between three points for workout monitoring (cached)."""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle if angle <= 180.0 else (360 - angle)

    @staticmethod
    def estimate_pose_angle(a: Any, b: Any, c: Any) -> float:
        """Calculate the angle between three points for workout monitoring.

        Args:
            a (Any): The coordinates of the first point (e.g. list/tuple/NumPy array/torch tensor).
            b (Any): The coordinates of the second point (vertex).
            c (Any): The coordinates of the third point.

        Returns:
            (float): The angle in degrees between the three points.
        """
        a_xy, b_xy, c_xy = (
            WorkflowAnnotator._point_xy(a),
            WorkflowAnnotator._point_xy(b),
            WorkflowAnnotator._point_xy(c),
        )
        return WorkflowAnnotator._estimate_pose_angle_cached(a_xy, b_xy, c_xy)

    def draw_specific_kpts(
        self,
        keypoints: list[list[float]],
        indices: list[int] | None = None,
        radius: int = 2,
        conf_thresh: float = 0.25,
    ) -> np.ndarray:
        """Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list[list[float]]): Keypoints data to be plotted, each in format [x, y, confidence].
            indices (list[int], optional): Keypoint indices to be plotted. The drawing order follows the order of this
                list.
            radius (int): Keypoint radius.
            conf_thresh (float): Confidence threshold for keypoints.

        Returns:
            (np.ndarray): Image with drawn keypoints.

        Notes:
            Keypoint format: [x, y] or [x, y, confidence].
            Modifies self.im in-place.
        """
        indices = indices or [2, 5, 7]
        n = len(keypoints)
        points = [
            (int(keypoints[j][0]), int(keypoints[j][1]))
            for j in indices
            if 0 <= j < n and (float(keypoints[j][2]) if len(keypoints[j]) > 2 else 1.0) >= conf_thresh
        ]

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw circles for keypoints
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(
        self,
        display_text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> int:
        """Draw workout text with a background on the image.

        Args:
            display_text (str): The text to be displayed.
            position (tuple[int, int]): Coordinates (x, y) on the image where the text will be placed.
            color (tuple[int, int, int]): Text background color.
            txt_color (tuple[int, int, int]): Text foreground color.

        Returns:
            (int): The height of the text.
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

        # Draw background rectangle
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # Draw text
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self,
        angle_text: str,
        count_text: str,
        stage_text: str,
        center_kpt: list[int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """Plot the pose angle, count value, and step stage for workout monitoring.

        Args:
            angle_text (str): Angle value for workout monitoring.
            count_text (str): Counts value for workout monitoring.
            stage_text (str): Stage decision for workout monitoring.
            center_kpt (list[int]): Centroid pose index for workout monitoring.
            color (tuple[int, int, int]): Text background color.
            txt_color (tuple[int, int, int]): Text foreground color.
        """
        # Format text
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # Draw angle, count and stage text
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(
        self,
        pixels_distance: float,
        centroids: list[tuple[int, int]],
        line_color: tuple[int, int, int] = (104, 31, 17),
        centroid_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """Plot the distance and line between two centroids on the frame.

        Args:
            pixels_distance (float): Pixel distance between two bounding-box centroids.
            centroids (list[tuple[int, int]]): Bounding box centroids data.
            line_color (tuple[int, int, int]): Distance line color.
            centroid_color (tuple[int, int, int]): Bounding box centroid color.
        """
        # Get the text size
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # Define corners with 10-pixel margin and draw rectangle
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # Calculate the position for the text with a 10-pixel margin and draw text
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(
        self,
        im0: np.ndarray,
        text: str,
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        x_center: float,
        y_center: float,
        margin: int,
    ):
        """Display the bounding boxes labels in parking management app.

        Args:
            im0 (np.ndarray): Inference image.
            text (str): Object/class name.
            txt_color (tuple[int, int, int]): Display color for text foreground.
            bg_color (tuple[int, int, int]): Display color for text background.
            x_center (float): The x position center point for bounding box.
            y_center (float): The y position center point for bounding box.
            margin (int): The gap between text and rectangle for better display.
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),  # Ensure color values are int
            -1,
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),  # Ensure color values are int
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def sweep_annotator(
        self,
        line_x: int = 0,
        line_y: int = 0,
        label: str | None = None,
        color: tuple[int, int, int] = (221, 0, 186),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """Draw a sweep annotation line and an optional label.

        Args:
            line_x (int): The x-coordinate of the sweep line.
            line_y (int): The y-coordinate limit of the sweep line.
            label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
            color (tuple[int, int, int]): BGR color for the line and label background (OpenCV format).
            txt_color (tuple[int, int, int]): BGR color for the label text (OpenCV format).
        """
        # Draw the sweep line
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # Draw label, if provided
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def visioneye(
        self,
        box: list[float],
        center_point: tuple[int, int],
        color: tuple[int, int, int] = (235, 219, 11),
        pin_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """Perform pinpoint human-vision eye mapping and plotting.

        Args:
            box (list[float]): Bounding box coordinates in format [x1, y1, x2, y2].
            center_point (tuple[int, int]): Center point for vision eye view.
            color (tuple[int, int, int]): Object centroid and line color.
            pin_color (tuple[int, int, int]): Visioneye point color.
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def adaptive_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        shape: str = "rect",
        margin: int = 5,
    ):
        """Draw a label with a background rectangle or circle centered within a given bounding box.

        Args:
            box (tuple[float, float, float, float]): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple[int, int, int]): The background color of the rectangle (B, G, R).
            txt_color (tuple[int, int, int]): The color of the text (B, G, R).
            shape (str): Label shape. Options: "circle" or "rect".
            margin (int): The margin between the text and the rectangle border.
        """
        if shape == "circle" and len(label) > 3:
            LOGGER.warning(f"Length of label is {len(label)}, only first 3 letters will be used for circle annotation.")
            label = label[:3]

        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # Bounding-box center
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]  # Get size of the text
        text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2  # Calculate top-left corner of text

        if shape == "circle":
            cv2.circle(
                self.im,
                (x_center, y_center),
                int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,  # Calculate the radius
                color,
                -1,
            )
        else:
            cv2.rectangle(
                self.im,
                (text_x - margin, text_y - text_size[1] - margin),  # Calculate coordinates of the rectangle
                (text_x + text_size[0] + margin, text_y + margin),  # Calculate coordinates of the rectangle
                color,
                -1,
            )

        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),  # Calculate top-left corner of the text
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )