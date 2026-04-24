"""Webcam capture handler for RT-MBAS."""
import cv2
import numpy as np

from app.config import WEBCAM_INDEX, TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT


class WebcamHandler:
    """Manages webcam capture via OpenCV VideoCapture."""

    def __init__(
        self,
        index: int = WEBCAM_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = TARGET_FPS,
    ):
        """Store capture parameters; does not open the device yet."""
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> bool:
        """
        Open the capture device and configure resolution and FPS.

        Returns True on success, False if the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print(
                f"Error: Cannot open camera at index {self.index}. "
                "Check WEBCAM_INDEX in app/config.py."
            )
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read one frame from the capture device.

        Returns (success, frame). Never raises — returns (False, None) on failure.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def get_fps(self) -> float:
        """Return the actual FPS reported by the capture device."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self) -> None:
        """Release the capture device and destroy all OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def __iter__(self):
        """Yield frames continuously; skips failures silently."""
        while True:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                continue
            yield frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.release()
