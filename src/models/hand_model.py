import cv2
import mediapipe as mp
import numpy as np

class HandFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features(self, frame):
        if frame is None:
            print("Warning: Input frame is None in HandFeatureExtractor")
            return np.zeros((2, 21, 3))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        all_hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                all_hand_landmarks.append(landmarks)

        if not all_hand_landmarks:
            print("Warning: No hands detected in HandFeatureExtractor")

        # If less than two hands are detected, pad with zeros
        while len(all_hand_landmarks) < 2:
            all_hand_landmarks.append(np.zeros((21, 3)))

        return np.array(all_hand_landmarks)