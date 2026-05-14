import logging
import cv2
import numpy as np
from src.translator import HandSignLanguageTranslator
from config.config import Config
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from collections import deque
import colorlog
import traceback
from tqdm import tqdm

# 配置日志
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 获取系统默认字体
font_path = fm.findfont(fm.FontProperties(family='SimSun'))

CONFIDENCE_THRESHOLD = 0.9  # 提高置信度阈值
MOTION_THRESHOLD = 0.020000
STATIC_THRESHOLD = 0.007000
STATIC_TIME = 1.0
MOTION_BUFFER_SIZE = 5
MAX_MOTION_THRESHOLD = 0.1


def put_chinese_text(img, text, position, font_path, font_size, color):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype(font_path, font_size)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error in put_chinese_text: {e}")
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img


def calculate_motion(prev_keypoints, curr_keypoints):
    if prev_keypoints is None or curr_keypoints is None:
        return 0

    if prev_keypoints.shape != curr_keypoints.shape:
        return 0

    if len(prev_keypoints.shape) == 1:
        prev_keypoints = prev_keypoints.reshape(-1, 1)
        curr_keypoints = curr_keypoints.reshape(-1, 1)

    valid_points = np.logical_and(np.any(prev_keypoints != 0, axis=1), np.any(curr_keypoints != 0, axis=1))
    if not np.any(valid_points):
        return 0

    diff = np.linalg.norm(curr_keypoints[valid_points] - prev_keypoints[valid_points], axis=1)

    weights = np.ones(len(diff))
    if len(weights) >= 5:
        weights[-5:] = 2

    weighted_motion = np.average(diff, weights=weights)
    return weighted_motion


def test_real_time_translation():
    config = Config()

    print("初始化翻译器...")
    with tqdm(total=100, desc="加载模型") as pbar:
        translator = HandSignLanguageTranslator(config)
        pbar.update(50)
        translator.start_capture()
        pbar.update(50)

    logger.info("Starting real-time translation...")
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    print("Press 'q' to quit")

    prev_keypoints = None
    motion_history = deque(maxlen=MOTION_BUFFER_SIZE)
    is_gesture_active = False
    gesture_translations = []
    static_start_time = None
    last_active_time = time.time()
    final_translation = ""
    top_5_translations = []

    try:
        while True:
            result = translator.translate_frame()
            if len(result) == 5:
                translation, vis_frame, max_prob, keypoints, top_5_translations = result
            elif len(result) == 4:
                translation, vis_frame, max_prob, keypoints = result
                top_5_translations = [(translation, max_prob)]
            else:
                logger.warning(f"Unexpected number of return values from translate_frame: {len(result)}")
                continue

            if vis_frame is not None and keypoints is not None:
                current_time = time.time()

                try:
                    motion = calculate_motion(prev_keypoints, keypoints)
                    prev_keypoints = keypoints.copy()
                    motion_history.append(motion)

                    avg_motion = np.mean(motion_history) if motion_history else 0

                    if avg_motion > MOTION_THRESHOLD:
                        if not is_gesture_active:
                            is_gesture_active = True
                            logger.info("New gesture started")
                        last_active_time = current_time
                        static_start_time = None
                    else:
                        if static_start_time is None:
                            static_start_time = current_time
                        elif current_time - static_start_time >= STATIC_TIME:
                            if is_gesture_active:
                                is_gesture_active = False
                                logger.info("Gesture ended due to inactivity")
                                if gesture_translations:
                                    final_translation = max(set(gesture_translations), key=gesture_translations.count)
                                    logger.info(f"Final translation: {final_translation}")
                                gesture_translations = []

                    # 更新状态和显示文本
                    if is_gesture_active:
                        status_color = (255, 0, 0)  # 蓝色
                        text = "识别中..."
                        if translation and max_prob >= CONFIDENCE_THRESHOLD:
                            gesture_translations.append(translation)
                    else:
                        status_color = (0, 0, 255)  # 红色
                        text = "等待手势..."

                    # 在右上角显示手势状态
                    cv2.circle(vis_frame, (vis_frame.shape[1] - 30, 30), 10, status_color, -1)

                    # 显示静止计时器
                    if static_start_time is not None:
                        static_time = current_time - static_start_time
                        cv2.putText(vis_frame, f"Static: {static_time:.1f}s", (10, vis_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 显示当前状态文本
                    vis_frame = put_chinese_text(vis_frame, text, (10, 30), font_path, 32, (255, 255, 255))

                    # 显示最终翻译结果
                    if final_translation:
                        vis_frame = put_chinese_text(vis_frame, f"最终翻译: {final_translation}", (10, 70), font_path,
                                                     32, (0, 255, 0))

                    # 显示当前动作幅度
                    cv2.putText(vis_frame, f"Motion: {avg_motion:.6f}", (10, vis_frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 显示 top 5 翻译结果
                    for i, (trans, prob) in enumerate(top_5_translations[:5]):
                        text = f"{i + 1}. {trans}: {prob:.2f}"
                        vis_frame = put_chinese_text(vis_frame, text, (10, 110 + i * 40), font_path, 24, (255, 255, 0))

                    cv2.imshow('Video', vis_frame)

                except Exception as e:
                    logger.error(f"Error in motion calculation: {e}")
                    logger.error(traceback.format_exc())

            else:
                logger.warning("No video frame or keypoints received")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  # Reduce delay to improve responsiveness

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        translator.stop_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_real_time_translation()