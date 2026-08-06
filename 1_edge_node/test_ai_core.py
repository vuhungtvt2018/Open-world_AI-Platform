import cv2

from packages.ai.engines.pytorch import UltralyticsBackend
from packages.ai.tasks.detection import DetectionTask


def get_bbox_coordinates(bbox) -> tuple[int, int, int, int]:
    """
    Chuyển bbox về định dạng:
    x1, y1, x2, y2

    Hỗ trợ bbox dạng:
    - list/tuple: [x1, y1, x2, y2]
    - dict
    - object có thuộc tính x1, y1, x2, y2
    """

    if isinstance(bbox, (list, tuple)):
        x1, y1, x2, y2 = bbox[:4]

    elif isinstance(bbox, dict):
        x1 = bbox.get("x1", bbox.get("xmin"))
        y1 = bbox.get("y1", bbox.get("ymin"))
        x2 = bbox.get("x2", bbox.get("xmax"))
        y2 = bbox.get("y2", bbox.get("ymax"))

    else:
        x1 = getattr(bbox, "x1", getattr(bbox, "xmin", None))
        y1 = getattr(bbox, "y1", getattr(bbox, "ymin", None))
        x2 = getattr(bbox, "x2", getattr(bbox, "xmax", None))
        y2 = getattr(bbox, "y2", getattr(bbox, "ymax", None))

    if None in (x1, y1, x2, y2):
        raise ValueError(f"Không xác định được định dạng bbox: {bbox}")

    return int(x1), int(y1), int(x2), int(y2)


def draw_predictions(image, predictions):
    """
    Vẽ bounding box, nhãn, confidence và tâm đối tượng.
    """

    output_image = image.copy()
    image_height, image_width = output_image.shape[:2]

    for prediction in predictions:
        x1, y1, x2, y2 = get_bbox_coordinates(prediction.bbox)

        # Giới hạn bbox nằm trong ảnh
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        confidence = float(prediction.confidence)
        label = f"{prediction.class_name}: {confidence:.2f}"

        # Vẽ bounding box
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )

        # Tính kích thước phần chữ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )

        # Đặt nhãn phía trên bbox; nếu sát mép trên thì đặt bên trong bbox
        label_y = y1 - 10

        if label_y - text_height < 0:
            label_y = y1 + text_height + 10

        background_top = label_y - text_height - 6
        background_bottom = label_y + baseline + 4

        # Nền cho nhãn
        cv2.rectangle(
            output_image,
            (x1, background_top),
            (x1 + text_width + 10, background_bottom),
            color=(0, 255, 0),
            thickness=-1,
        )

        # Vẽ tên class và confidence
        cv2.putText(
            output_image,
            label,
            (x1 + 5, label_y),
            font,
            font_scale,
            color=(0, 0, 0),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

        # Vẽ tâm đối tượng
        center = prediction.center

        if center is not None:
            center_x = int(center[0])
            center_y = int(center[1])
        else:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

        cv2.circle(
            output_image,
            (center_x, center_y),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    return output_image


def main() -> None:
    """
    06082026 - KIET
    Kiểm tra DetectionTask với UltralyticsBackend.
    Hiển thị bbox, nhãn và confidence trên ảnh.
    """

    backend = UltralyticsBackend(
        checkpoint=(r"C:\Users\IVS-1437\Desktop\PlatformAI\Open-world_AI-Platform\bulong_8ly_s.pt"),
        device="cpu",
        confidence=0.25,
    )

    try:
        backend.load()

        task = DetectionTask(
            backend=backend,
        )

        image_path = (r"C:\Users\IVS-1437\Desktop\PlatformAI\Open-world_AI-Platform\IMG_20250904_171011_482283.jpg")

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(
                f"Cannot read test image: {image_path}"
            )

        result = task.run(image)

        print(f"Objects: {len(result.predictions)}")
        print(f"Type: {result.task.value}")

        for prediction in result.predictions:
            print({
                "class": prediction.class_name,
                "confidence": prediction.confidence,
                "bbox": prediction.bbox,
                "center": prediction.center,
            })

        # Vẽ kết quả detection
        output_image = draw_predictions(
            image=image,
            predictions=result.predictions,
        )

        # Lưu ảnh kết quả
        output_path = "detection_result.jpg"

        success = cv2.imwrite(
            output_path,
            output_image,
        )

        if success:
            print(f"Saved detection result: {output_path}")
        else:
            print("Cannot save detection result")

        # Thu nhỏ ảnh nếu ảnh quá lớn
        display_image = output_image.copy()

        max_display_width = 1280
        max_display_height = 800

        height, width = display_image.shape[:2]

        scale = min(
            max_display_width / width,
            max_display_height / height,
            1.0,
        )

        if scale < 1.0:
            display_image = cv2.resize(
                display_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        # Hiển thị ảnh
        cv2.imshow(
            "Detection Result",
            display_image,
        )

        print("Press any key to close the image window.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        backend.close()


if __name__ == "__main__":
    main()