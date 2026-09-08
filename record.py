import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue

import numpy as np
import sounddevice as sd

from database import create_session, finish_session, insert_samples

logger = logging.getLogger(__name__)

DEFAULT_RATE = 16000
DEFAULT_CHUNK = 1024
DEFAULT_THRESHOLD = 3000
INT16_SCALE = 32768.0

# Bark detection frequency band (Hz) - typical dog bark range
BARK_LOW_HZ = 400
BARK_HIGH_HZ = 2000
# Minimum energy ratio in bark band vs total to count as bark (0-1)
BARK_ENERGY_RATIO = 0.35


def _detect_bark_fft(audio_np, sample_rate):
    """Return (is_bark: int, bark_confidence: float) based on FFT frequency analysis.

    Dog barks typically concentrate energy in 400-2000 Hz.
    We compare energy in that band vs total signal energy.
    """
    if audio_np.size < 128:
        return 0, 0.0

    # Flatten to mono
    mono = audio_np.mean(axis=1) if audio_np.ndim > 1 else audio_np

    # Window + FFT
    window = np.hanning(len(mono))
    fft = np.fft.rfft(mono * window)
    power = np.abs(fft) ** 2

    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)

    total_energy = np.sum(power)
    if total_energy == 0:
        return 0, 0.0

    bark_mask = (freqs >= BARK_LOW_HZ) & (freqs <= BARK_HIGH_HZ)
    bark_energy = np.sum(power[bark_mask])
    confidence = bark_energy / total_energy

    is_bark = 1 if confidence >= BARK_ENERGY_RATIO else 0
    return is_bark, confidence


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int


def list_input_devices():
    devices = []
    for index, info in enumerate(sd.query_devices()):
        max_input_channels = int(info.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        devices.append(
            InputDevice(
                index=index,
                name=str(info.get("name", f"Input {index}")),
                max_input_channels=max_input_channels,
                default_sample_rate=int(info.get("default_samplerate", DEFAULT_RATE)),
            )
        )
    return devices


def compute_rms_volume(samples):
    audio_np = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio_np))) * INT16_SCALE)


class Recorder:
    def __init__(self, threshold=DEFAULT_THRESHOLD, rate=DEFAULT_RATE, chunk=DEFAULT_CHUNK):
        self.threshold = threshold
        self.rate = rate
        self.chunk = chunk
        self.recording_flag = False
        self.thread = None
        self.last_error = None
        self.selected_device_index = None
        self.device_name = ""
        self.latest_volume = 0.0
        self.bark_confidence = 0.0
        self.bark_count = 0
        self.sample_count = 0
        self.started_at = None
        self.session_id = None
        self._lock = threading.Lock()
        self._sample_queue = Queue()
        self._db_thread = None

    def _db_writer(self, session_id):
        buffer = []
        while self.recording_flag or not self._sample_queue.empty():
            try:
                item = self._sample_queue.get(timeout=0.5)
                buffer.append(item)
                if len(buffer) >= 50:
                    insert_samples(session_id, buffer)
                    buffer = []
            except BaseException:
                logger.exception("Error in _db_writer")
        if buffer:
            insert_samples(session_id, buffer)

    def _record(self, device_index, channels):
        stream = None
        try:
            self.started_at = time.time()
            self.session_id = create_session(device_name=self.device_name, threshold=self.threshold)
            self._db_thread = threading.Thread(
                target=self._db_writer, args=(self.session_id,), daemon=True
            )
            self._db_thread.start()

            def callback(indata, frames, stream_time, status):
                del frames, stream_time
                if status:
                    self.last_error = str(status)

                volume = compute_rms_volume(indata)
                is_bark_fft, bark_confidence = _detect_bark_fft(indata, self.rate)

                # Bark = loud AND has frequency signature of a dog bark
                is_bark = int((volume > self.threshold) and (is_bark_fft == 1))

                with self._lock:
                    self.latest_volume = volume
                    self.sample_count += 1
                    self.bark_count += is_bark
                    self.bark_confidence = bark_confidence

                self._sample_queue.put((self.session_id, time.time(), volume, is_bark))

            stream = sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=self.rate,
                blocksize=self.chunk,
                dtype="float32",
                callback=callback,
            )
            stream.start()

            while self.recording_flag:
                time.sleep(max(self.chunk / self.rate, 0.05))
        except BaseException as exc:
            logger.exception("Error in _record")
            self.last_error = str(exc)
        finally:
            self.recording_flag = False
            if self._db_thread is not None:
                self._db_thread.join(timeout=3)
            if stream is not None:
                stream.stop()
                stream.close()
            if self.session_id is not None:
                finish_session(
                    self.session_id,
                    sample_count=self.sample_count,
                    bark_count=self.bark_count,
                    peak_volume=self.latest_volume,
                )

    def start(self, device_index, device_name="", channels=1, rate=None):
        if self.recording_flag:
            return False

        self.last_error = None
        self.latest_volume = 0.0
        self.bark_count = 0
        self.sample_count = 0
        self.selected_device_index = device_index
        self.device_name = device_name
        if rate is not None:
            self.rate = rate
        self.recording_flag = True
        self.thread = threading.Thread(
            target=self._record, args=(device_index, channels), daemon=True
        )
        self.thread.start()
        return True

    def stop(self):
        if not self.recording_flag:
            return False
        self.recording_flag = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        return True

    def get_status(self):
        with self._lock:
            return {
                "recording": self.recording_flag,
                "latest_volume": self.latest_volume,
                "bark_confidence": self.bark_confidence,
                "bark_count": self.bark_count,
                "sample_count": self.sample_count,
                "last_error": self.last_error,
                "device_index": self.selected_device_index,
                "session_id": self.session_id,
            }
