import cv2
import mediapipe as mp
import numpy as np
import torch

# MediaPipe初始化
mp_holistic = mp.solutions.holistic


def map_mediapipe_to_kinect(landmarks):
    kinect_landmarks = np.zeros((25, 3))

    # 映射MediaPipe关键点到Kinect关键点
    kinect_landmarks[0] = landmarks[23]  # SpineBase
    kinect_landmarks[1] = landmarks[24]  # SpineMid
    kinect_landmarks[2] = landmarks[12]  # Neck
    kinect_landmarks[3] = landmarks[0]  # Head
    kinect_landmarks[4] = landmarks[11]  # ShoulderLeft
    kinect_landmarks[5] = landmarks[13]  # ElbowLeft
    kinect_landmarks[6] = landmarks[15]  # WristLeft
    kinect_landmarks[7] = landmarks[19]  # HandLeft
    kinect_landmarks[8] = landmarks[12]  # ShoulderRight
    kinect_landmarks[9] = landmarks[14]  # ElbowRight
    kinect_landmarks[10] = landmarks[16]  # WristRight
    kinect_landmarks[11] = landmarks[20]  # HandRight
    kinect_landmarks[12] = landmarks[23]  # HipLeft
    kinect_landmarks[13] = landmarks[25]  # KneeLeft
    kinect_landmarks[14] = landmarks[27]  # AnkleLeft
    kinect_landmarks[15] = landmarks[31]  # FootLeft
    kinect_landmarks[16] = landmarks[24]  # HipRight
    kinect_landmarks[17] = landmarks[26]  # KneeRight
    kinect_landmarks[18] = landmarks[28]  # AnkleRight
    kinect_landmarks[19] = landmarks[32]  # FootRight
    kinect_landmarks[20] = landmarks[11]  # SpineShoulder (approximation)
    kinect_landmarks[21] = landmarks[19]  # HandTipLeft
    kinect_landmarks[22] = landmarks[21]  # ThumbLeft
    kinect_landmarks[23] = landmarks[20]  # HandTipRight
    kinect_landmarks[24] = landmarks[22]  # ThumbRight

    return kinect_landmarks


def extract_keypoints(video_path):
    """
    从视频中提取骨骼关键点并映射到Kinect格式
    """
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark])
                kinect_landmarks = map_mediapipe_to_kinect(landmarks)
                keypoints.append(kinect_landmarks.flatten())
            else:
                keypoints.append(np.zeros(25 * 3))

    cap.release()
    return keypoints


def format_keypoints(keypoints, target_frames=36, target_points=25):
    # 调整帧数
    if len(keypoints) > target_frames:
        indices = np.linspace(0, len(keypoints) - 1, target_frames, dtype=int)
        keypoints = [keypoints[i] for i in indices]
    else:
        keypoints = keypoints + [keypoints[-1]] * (target_frames - len(keypoints))

    # 确保每帧都有正确数量的关键点
    formatted_keypoints = []
    for frame in keypoints:
        if len(frame) != target_points * 3:
            frame = np.zeros(target_points * 3)
        formatted_keypoints.append(frame)

    return np.array(formatted_keypoints)


def abs2rel(coords, crop_size):
    return coords / crop_size


def preprocess_keypoints(keypoints, crop_size=256):
    for i in range(len(keypoints)):
        keypoints[i] = abs2rel(keypoints[i], crop_size)
    return keypoints


def video_to_model_input(video_path, target_frames=36, target_points=25, crop_size=256):
    """
    将视频转换为模型输入
    """
    # 提取关键点
    raw_keypoints = extract_keypoints(video_path)

    # 格式化关键点
    formatted_keypoints = format_keypoints(raw_keypoints, target_frames, target_points)

    # 预处理
    preprocessed_keypoints = preprocess_keypoints(formatted_keypoints, crop_size)

    # 转换为模型输入格式
    model_input = torch.from_numpy(preprocessed_keypoints).float().unsqueeze(0)

    return model_input


def predict(video_path, model):
    """
    使用模型对视频进行预测
    """
    model_input = video_to_model_input(video_path)
    with torch.no_grad():
        prediction = model(model_input)
    return process_prediction(prediction)


def process_prediction(prediction):
    """
    处理模型的原始预测结果
    """
    # 这里需要根据您的模型输出格式来实现具体的处理逻辑
    # 例如，如果是分类任务，可能需要找出概率最大的类别
    probabilities = torch.nn.functional.softmax(prediction, dim=1)
    max_prob, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), max_prob.item()


# 使用示例
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"

    # 假设您已经加载了模型
    # model = load_your_model()

    # 生成模型输入
    model_input = video_to_model_input(video_path)
    print(f"Model input shape: {model_input.shape}")

    # 如果要保存为npy文件
    # np.save('output.npy', model_input.numpy())