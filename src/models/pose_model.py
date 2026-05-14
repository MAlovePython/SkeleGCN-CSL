import cv2
import mediapipe as mp
import numpy as np


class PoseFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    def extract_features(self, frame):
        if frame is None:
            return np.zeros((33, 3))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

        if not landmarks:
            return np.zeros((33, 3))

        return np.array(landmarks)