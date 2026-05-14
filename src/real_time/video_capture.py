import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VideoCapture:
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera with index {camera_index}")
            raise ValueError(f"Unable to open camera with index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        logger.info(f"Camera initialized with resolution {resolution} and FPS {fps}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            logger.debug(f"Frame read successfully. Shape: {frame.shape}")
            return self.preprocess_frame(frame)
        logger.warning("Failed to read frame from camera")
        return None

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.debug(f"Frame preprocessed. Shape: {frame.shape}")
        return frame.astype(np.uint8)

    def release(self):
        self.cap.release()
        logger.info("Camera released")