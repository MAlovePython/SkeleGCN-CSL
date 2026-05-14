import threading
from queue import Queue
import time
import logging

import numpy as np

from src.real_time.video_capture import VideoCapture
from src.real_time.audio_capture import AudioCapture

logger = logging.getLogger(__name__)

class RealTimePreprocessor:
    def __init__(self):
        self.video_capture = VideoCapture()
        self.audio_capture = AudioCapture(sample_rate=48000, channels=4, bit_depth=16)
        self.video_queue = Queue(maxsize=10)
        self.audio_queue = Queue(maxsize=10)
        self.is_running = False
        self.audio_buffer = np.array([])

    def start(self):
        self.is_running = True
        self.video_thread = threading.Thread(target=self._video_thread)
        self.audio_thread = threading.Thread(target=self._audio_thread)
        self.video_thread.start()
        self.audio_thread.start()
        logger.info("RealTimePreprocessor started")

    def _video_thread(self):
        logger.info("Video thread started")
        while self.is_running:
            frame = self.video_capture.read_frame()
            if frame is not None:
                if self.video_queue.full():
                    self.video_queue.get()  # 如果队列满了，移除最旧的帧
                self.video_queue.put(frame)
                logger.debug(f"Video frame added to queue. Queue size: {self.video_queue.qsize()}")
            else:
                logger.warning("Failed to read video frame")
            time.sleep(0.01)

    def _audio_thread(self):
        logger.info("Audio thread started")
        while self.is_running:
            audio = self.audio_capture.read_audio()
            if audio is not None:
                self.audio_buffer = np.concatenate((self.audio_buffer, audio.flatten()))
                if len(self.audio_buffer) >= 16000:  # 假设我们想要1秒的音频数据
                    if self.audio_queue.full():
                        self.audio_queue.get()
                    self.audio_queue.put(self.audio_buffer[:16000])
                    self.audio_buffer = self.audio_buffer[16000:]
                    logger.debug(f"Audio data added to queue. Queue size: {self.audio_queue.qsize()}")
            else:
                logger.warning("Failed to read audio data")
            time.sleep(0.01)

    def get_data(self):
        video = None if self.video_queue.empty() else self.video_queue.get()
        audio = None if self.audio_queue.empty() else self.audio_queue.get()
        if video is None:
            logger.warning("Video queue is empty")
        if audio is None:
            logger.warning("Audio queue is empty")
        return video, audio

    def stop(self):
        logger.info("Stopping RealTimePreprocessor")
        self.is_running = False
        self.video_thread.join()
        self.audio_thread.join()
        self.video_capture.release()
        self.audio_capture.close()
        logger.info("RealTimePreprocessor stopped")