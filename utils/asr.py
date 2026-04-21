import json
import os
import tempfile

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # int8 quantization keeps CPU memory and latency low
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_batch(wav_bytes: bytes, language: str = "ja") -> str:
    """Transcribe WAV bytes using faster-whisper. Downloads model on first call."""
    model = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path, language=language, beam_size=5)
        return "".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


class VoskRealtimeASR:
    """Streaming ASR using Vosk. Feed 30ms PCM chunks; get partial/final text."""

    def __init__(self, model_path: str = "vosk-model-small-ja-0.22", sample_rate: int = 16000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                "Download the Japanese small model:\n"
                "  wget https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip\n"
                "  unzip vosk-model-small-ja-0.22.zip\n"
                "Or set VOSK_MODEL_PATH to your model directory."
            )
        from vosk import KaldiRecognizer, Model
        self._rec = KaldiRecognizer(Model(model_path), sample_rate)

    def process(self, chunk: bytes) -> tuple[str, bool]:
        """Feed one audio chunk. Returns (text, is_final).

        is_final=True means Vosk completed an internal segment (not end-of-turn).
        Use flush() to finalize an utterance on VAD-detected silence.
        """
        if self._rec.AcceptWaveform(chunk):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False

    def flush(self) -> str:
        """Finalize the current utterance and reset. Call on end-of-speech."""
        result = json.loads(self._rec.FinalResult())
        return result.get("text", "").strip()
