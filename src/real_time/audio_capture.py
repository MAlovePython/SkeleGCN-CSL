import pyaudio
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, sample_rate=48000, chunk_size=1024, channels=4, bit_depth=16):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.bit_depth = bit_depth
        self.p = pyaudio.PyAudio()

        # 根据位深度选择合适的格式
        if bit_depth == 16:
            self.format = pyaudio.paInt16
        elif bit_depth == 24:
            self.format = pyaudio.paInt24
        elif bit_depth == 32:
            self.format = pyaudio.paFloat32
        else:
            raise ValueError("Unsupported bit depth")

        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)

    def read_audio(self):
        try:
            data = self.stream.read(self.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16 if self.bit_depth <= 16 else np.int32)
            return self.preprocess_audio(audio_data)
        except IOError as e:
            logger.warning(f"Failed to read audio: {e}")
            return None

    def preprocess_audio(self, audio_data):
        # 将多通道数据重塑为 2D 数组
        audio_data = audio_data.reshape(-1, self.channels)

        # 如果需要，可以在这里添加更多的预处理步骤
        # 例如，可以选择特定的通道，或者对多个通道进行平均

        # 将数据标准化到 [-1, 1] 范围
        audio_data = audio_data.astype(np.float32) / (2 ** (self.bit_depth - 1))

        return audio_data

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()