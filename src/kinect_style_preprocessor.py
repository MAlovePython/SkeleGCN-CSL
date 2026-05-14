import cv2
import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import torch
import logging
from queue import Queue
import threading
import time
from src.models.hand_model import HandFeatureExtractor
from src.models.pose_model import PoseFeatureExtractor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KinectStylePreprocessor:
    def __init__(self, config):
        self.config = config
        self.pose_extractor = PoseFeatureExtractor()
        self.hand_extractor = HandFeatureExtractor()
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_frames = getattr(config, 'num_frames', 36)
        self.num_joints = 25  # Kinect风格的关键点数量
        self.crop_size = getattr(config, 'crop_size', 256)
        self.input_dim = getattr(config, 'input_dim', 3)

        self.video_queue = Queue(maxsize=5)
        self.is_running = False
        self.sliding_window = []
        self.process_every_n_frames = 3
        self.frame_count = 0

        logger.info("Opening camera")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera. Please check your camera connection.")
        logger.info("Camera opened successfully")

    def start_capture(self):
        logger.info("Starting video capture")
        if not self.cap.isOpened():
            logger.info("Reopening camera")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to reopen camera")
                raise RuntimeError("Failed to open camera. Please check your camera connection.")
        self.is_running = True
        self.video_thread = threading.Thread(target=self._video_thread)
        self.video_thread.start()

        logger.info("Warming up camera...")
        for _ in range(30):
            ret, _ = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame during warm-up")
            time.sleep(0.1)

        logger.info("Video capture started")

    def stop_capture(self):
        logger.info("Stopping video capture")
        self.is_running = False
        if self.video_thread.is_alive():
            self.video_thread.join()
        if self.cap.isOpened():
            self.cap.release()
        logger.info("Video capture stopped")

    def _video_thread(self):
        logger.info("Video thread started")
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    if self.frame_count % self.process_every_n_frames == 0:
                        if self.video_queue.full():
                            logger.debug("Video queue full, removing oldest frame")
                            self.video_queue.get()
                        self.video_queue.put(frame)
                        logger.debug(f"Video frame added to queue. Queue size: {self.video_queue.qsize()}")
                else:
                    logger.warning("Failed to read video frame")
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in video thread: {e}")
        logger.info("Video thread stopped")

    def map_mediapipe_to_kinect(self, pose_landmarks, hand_landmarks):
        kinect_landmarks = np.zeros((self.num_joints, 3))

        # 映射姿势关键点
        pose_mapping = {
            0: 3,  # nose -> Head
            11: 4,  # left_shoulder -> ShoulderLeft
            13: 5,  # left_elbow -> ElbowLeft
            15: 6,  # left_wrist -> WristLeft
            12: 8,  # right_shoulder -> ShoulderRight
            14: 9,  # right_elbow -> ElbowRight
            16: 10,  # right_wrist -> WristRight
            23: 12,  # left_hip -> HipLeft
            25: 13,  # left_knee -> KneeLeft
            27: 14,  # left_ankle -> AnkleLeft
            24: 16,  # right_hip -> HipRight
            26: 17,  # right_knee -> KneeRight
            28: 18,  # right_ankle -> AnkleRight
        }

        detected_points = 0
        for mp_idx, kinect_idx in pose_mapping.items():
            if pose_landmarks[mp_idx][0] != 0 or pose_landmarks[mp_idx][1] != 0:
                kinect_landmarks[kinect_idx] = pose_landmarks[mp_idx]
                detected_points += 1
                logger.debug(f"Mapped pose joint {mp_idx} to Kinect joint {kinect_idx}: {kinect_landmarks[kinect_idx]}")

        # 估算额外的姿势关键点
        if detected_points >= 4:
            # SpineBase (0)
            kinect_landmarks[0] = (kinect_landmarks[12] + kinect_landmarks[16]) / 2
            # SpineMid (1) - 修改为使用肩部和臀部的中点
            shoulder_center = (kinect_landmarks[4] + kinect_landmarks[8]) / 2
            hip_center = (kinect_landmarks[12] + kinect_landmarks[16]) / 2
            kinect_landmarks[1] = (shoulder_center + hip_center) / 2
            # Neck (2) - 使用头部和肩膀的中点
            kinect_landmarks[2] = (kinect_landmarks[3] + (kinect_landmarks[4] + kinect_landmarks[8]) / 2) / 2
            # SpineShoulder (20)
            kinect_landmarks[20] = (kinect_landmarks[4] + kinect_landmarks[8]) / 2
            detected_points += 4

        # 映射手部关键点（保持不变）
        left_hand_detected = hand_landmarks[0].size > 0
        right_hand_detected = hand_landmarks[1].size > 0

        if left_hand_detected:
            kinect_landmarks[7] = hand_landmarks[0][9]  # HandLeft (使用掌心点)
            kinect_landmarks[21] = hand_landmarks[0][8]  # HandTipLeft (使用食指尖)
            kinect_landmarks[22] = hand_landmarks[0][4]  # ThumbLeft
            detected_points += 3

        if right_hand_detected:
            kinect_landmarks[11] = hand_landmarks[1][9]  # HandRight (使用掌心点)
            kinect_landmarks[23] = hand_landmarks[1][8]  # HandTipRight (使用食指尖)
            kinect_landmarks[24] = hand_landmarks[1][4]  # ThumbRight
            detected_points += 3

        # 如果只检测到一只手，确保使用正确的索引
        if left_hand_detected and not right_hand_detected:
            kinect_landmarks[11] = np.zeros(3)  # 清除右手数据
            kinect_landmarks[23] = np.zeros(3)
            kinect_landmarks[24] = np.zeros(3)
        elif right_hand_detected and not left_hand_detected:
            kinect_landmarks[7] = np.zeros(3)  # 清除左手数据
            kinect_landmarks[21] = np.zeros(3)
            kinect_landmarks[22] = np.zeros(3)

        logger.debug(f"Detected {detected_points} points out of {self.num_joints} Kinect joints")
        return kinect_landmarks, detected_points

    def preprocess_frame(self, frame):
        try:
            pose_landmarks = self.pose_extractor.extract_features(frame)
            hand_landmarks = self.hand_extractor.extract_features(frame)

            kinect_landmarks, detected_points = self.map_mediapipe_to_kinect(pose_landmarks, hand_landmarks)

            self.last_keypoints = kinect_landmarks

            logger.debug("Before coordinate adjustment:")
            for i, (x, y, z) in enumerate(kinect_landmarks):
                if x != 0 or y != 0:
                    logger.debug(f"Joint {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

            # 调整坐标系统
            kinect_landmarks[:, 0] = np.where(kinect_landmarks[:, 0] != 0, (kinect_landmarks[:, 0] - 0.5) * 2, -1)
            kinect_landmarks[:, 1] = np.where(kinect_landmarks[:, 1] != 0, (kinect_landmarks[:, 1] - 0.5) * 2, -1)

            logger.debug("After coordinate adjustment:")
            for i, (x, y, z) in enumerate(kinect_landmarks):
                logger.debug(f"Joint {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

            logger.debug(f"Number of detected keypoints: {detected_points}")

            return kinect_landmarks, self.visualize_keypoints(frame, kinect_landmarks), detected_points
        except Exception as e:
            logger.error(f"Error in preprocess_frame: {e}")
            return np.zeros((self.num_joints, 3)), frame, 0

    def visualize_keypoints(self, frame, landmarks):
        h, w, _ = frame.shape
        logger.debug(f"Frame dimensions: width={w}, height={h}")
        for i, (x, y, z) in enumerate(landmarks):
            if x != -1 and y != -1:  # 只绘制有效点
                x_pixel = int((x + 1) / 2 * w)  # 将 x 从 [-1, 1] 映射回 [0, w]
                y_pixel = int((y + 1) / 2 * h)  # 将 y 从 [-1, 1] 映射回 [0, h]
                logger.debug(f"Drawing joint {i} at pixel coordinates: x={x_pixel}, y={y_pixel}")
                cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x_pixel + 5, y_pixel + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    def check_keypoints(self, keypoints, detected_points):
        logger.debug(f"Number of detected keypoints: {detected_points}/{self.num_joints}")
        return detected_points >= 4  # 至少检测到4个点（例如头部和肩膀）

    def is_valid_pose(self, keypoints):
        head = keypoints[3]
        left_shoulder = keypoints[4]
        right_shoulder = keypoints[8]

        if np.all(head == -1) or np.all(left_shoulder == -1) or np.all(right_shoulder == -1):
            logger.warning("Missing critical points for pose validation")
            return False

        logger.debug(f"Head Y: {head[1]:.4f}, Left Shoulder Y: {left_shoulder[1]:.4f}, Right Shoulder Y: {right_shoulder[1]:.4f}")

        if abs(head[1] - left_shoulder[1]) < 0.1 or abs(head[1] - right_shoulder[1]) < 0.1:
            logger.warning("Invalid pose: Head too close to shoulders")
            return False

        shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
        logger.debug(f"Shoulder distance: {shoulder_distance:.4f}")
        if shoulder_distance < 0.2:  # 增加这个阈值
            logger.warning("Invalid pose: Shoulders are too close")
            return False

        return True

    def get_model_input(self):
        if self.video_queue.empty():
            logger.warning("Video queue is empty")
            return None, None

        frame = self.video_queue.get()
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error(f"Invalid frame type: {type(frame)}")
            return None, None

        start_time = time.time()
        keypoints, vis_frame, detected_points = self.preprocess_frame(frame)
        logger.debug(f"Preprocessing time: {time.time() - start_time:.4f}s")

        if keypoints is None or not isinstance(keypoints, np.ndarray):
            logger.error(f"Invalid keypoints type: {type(keypoints)}")
            return None, vis_frame

        if not self.check_keypoints(keypoints, detected_points):
            logger.warning("Insufficient keypoints detected")
            return "Incomplete pose detection", vis_frame

        if not self.is_valid_pose(keypoints):
            logger.warning("Invalid pose detected")
            return "Invalid pose", vis_frame

        self.sliding_window.append(keypoints)
        if len(self.sliding_window) > self.target_frames:
            self.sliding_window.pop(0)

        if len(self.sliding_window) < self.target_frames:
            logger.debug(f"Collecting data: {len(self.sliding_window)}/{self.target_frames}")
            return "Collecting data", vis_frame

        model_input = torch.from_numpy(np.array(self.sliding_window)).float()
        model_input = model_input.view(1, self.target_frames, self.num_joints, -1)

        logger.debug(f"Model input shape: {model_input.shape}")

        return model_input, vis_frame

    def get_last_keypoints(self):
        return self.last_keypoints if hasattr(self, 'last_keypoints') else None

def main():
    class Config:
        def __init__(self):
            self.num_frames = 36
            self.crop_size = 256
            self.input_dim = 3

    config = Config()
    preprocessor = KinectStylePreprocessor(config)
    preprocessor.start_capture()

    try:
        while True:
            model_input, vis_frame = preprocessor.get_model_input()
            if isinstance(model_input, torch.Tensor):
                logger.info(f"Got model input with shape: {model_input.shape}")
            elif isinstance(model_input, str):
                logger.info(f"Status: {model_input}")

            if vis_frame is not None:
                cv2.imshow('Kinect Style Keypoints', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        preprocessor.stop_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()