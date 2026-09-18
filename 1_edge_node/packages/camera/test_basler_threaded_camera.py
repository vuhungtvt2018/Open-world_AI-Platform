import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(r"D:\ivs\my_folder\freelance\visual_inspection\sprint_20250908_20250912_direction\demo_bulong\1_edge_node")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.camera import Basler_Threaded_Camera


if __name__ == "__main__":
    cam = None
    try:
        print("Initializing Basler_Threaded_Camera...")
        cam = Basler_Threaded_Camera(
            exposure_time_us=5000,
            gain=10,
            width=3840,
            height=2748,
            auto_resolution=False,
        )

        print("Camera started. Press 'q' to quit.")
        frame_count = 0

        while True:
            frame = cam.read()
            print(f"Frame {frame_count}: shape={frame.shape if frame is not None else None}, dtype={frame.dtype if frame is not None else None}")
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            # frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Basler Live Stream", frame)
            cv2.imwrite(f"frame_{frame_count:04d}.png", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        print(f"Test completed. Read {frame_count} frames.")

    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        cv2.destroyAllWindows()
        if cam is not None:
            cam.stop()
            print("Camera stopped.")
