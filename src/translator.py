import torch
import logging
from src.models.csl500_model import IntermediateCSL500Model
from .kinect_style_preprocessor import KinectStylePreprocessor
import traceback
import time
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandSignLanguageTranslator:
    def __init__(self, config):
        logger.info("Initializing HandSignLanguageTranslator")
        self.config = config
        self.csl500_model = IntermediateCSL500Model(config)
        self.preprocessor = KinectStylePreprocessor(config)

        self.num_frames = getattr(config, 'num_frames', 36)
        self.num_joints = config.num_joints
        self.input_dim = config.input_dim

        self.label_map = config.label_map

        self.needed_joints = [3, 2, 20, 1, 0, 4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24]

        try:
            logger.info(f"Loading model from {config.model_path}")
            checkpoint = torch.load(config.model_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.csl500_model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.csl500_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.csl500_model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Failed to load the model. Please check the model path and format.")

        self.csl500_model.eval()

    def start_capture(self):
        self.preprocessor.start_capture()

    def stop_capture(self):
        self.preprocessor.stop_capture()

    def translate_frame(self):
        logger.debug("Starting translate_frame method")
        try:
            model_input, vis_frame = self.preprocessor.get_model_input()

            if isinstance(model_input, str):
                logger.debug(f"Received string input: {model_input}")
                return model_input, vis_frame, 0.0, None

            if model_input is None:
                logger.warning("Failed to get model input")
                return "No input data", vis_frame, 0.0, None

            logger.debug(f"Original model input shape: {model_input.shape}")

            # 选择需要的关节点
            model_input = model_input[:, :, self.needed_joints, :2]  # 只取x和y坐标
            logger.debug(f"Adjusted model input shape: {model_input.shape}")

            # Reshape input to match model expectations
            batch_size, seq_len, num_joints, coords = model_input.shape
            model_input = model_input.view(batch_size, seq_len, num_joints * coords)

            # 如果 num_joints * coords 不等于 17，我们需要进行调整
            if num_joints * coords != 17:
                logger.warning(f"Expected 17 features, but got {num_joints * coords}. Adjusting...")
                if num_joints * coords > 17:
                    model_input = model_input[:, :, :17]
                else:
                    pad_size = 17 - (num_joints * coords)
                    model_input = F.pad(model_input, (0, pad_size))

            logger.debug(f"Final model input shape: {model_input.shape}")

            # 确保输入张量的数据类型是float
            model_input = model_input.float()

            # 创建一个虚拟的lengths张量
            lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long)

            start_time = time.time()
            with torch.no_grad():
                translation = self.csl500_model(model_input, lengths)
            logger.debug(f"Model inference time: {time.time() - start_time:.4f}s")
            logger.debug(f"Translation shape: {translation.shape}")

            translation, max_prob = self._postprocess_translation(translation)

            # 获取原始关键点
            original_keypoints = self.preprocessor.get_last_keypoints()
            logger.debug(
                f"Original keypoints shape: {original_keypoints.shape if original_keypoints is not None else None}")

            return translation, vis_frame, max_prob, original_keypoints

        except Exception as e:
            logger.error(f"Error in translate_frame: {e}")
            logger.error(traceback.format_exc())
            return None, None, 0.0, None

    def _postprocess_translation(self, translation):
        try:
            probabilities = torch.nn.functional.softmax(translation, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            logger.debug(
                f"Top 5 predictions: {[(self.label_map[idx.item()], prob.item()) for idx, prob in zip(top5_idx[0], top5_prob[0])]}")

            max_prob, predicted = torch.max(probabilities, dim=1)
            logger.debug(f"Max probability: {max_prob[0].item():.4f}, Predicted class: {predicted[0].item()}")

            if max_prob.item() < 0.5:
                return "Unrecognized", max_prob.item()

            return self.label_map[predicted.item()], max_prob.item()
        except Exception as e:
            logger.error(f"Error in _postprocess_translation: {e}")
            logger.error(traceback.format_exc())
            return "Error in translation", 0.0

if __name__ == "__main__":
    import cv2
    from config.config import Config

    config = Config()
    translator = HandSignLanguageTranslator(config)
    translator.start_capture()

    try:
        while True:
            translation, vis_frame, max_prob = translator.translate_frame()
            print(f"Translation: {translation}, Probability: {max_prob:.4f}")

            if vis_frame is not None:
                text = f"{translation} ({max_prob:.2f})" if translation else "Collecting data"
                cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Hand Sign Language Translation', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)  # Add small delay to avoid excessive CPU usage
    finally:
        translator.stop_capture()
        cv2.destroyAllWindows()