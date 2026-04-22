import contextlib
import io
import os
import wave

import pyaudio
import webrtcvad

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples = 30ms
BYTES_PER_SAMPLE = 2  # int16


@contextlib.contextmanager
def _suppress_alsa_errors():
    """Redirect C-level stderr to /dev/null to silence ALSA/JACK noise."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


def _open_pyaudio() -> pyaudio.PyAudio:
    with _suppress_alsa_errors():
        pa = pyaudio.PyAudio()
    input_devices = [
        pa.get_device_info_by_index(i)
        for i in range(pa.get_device_count())
        if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
    ]
    if not input_devices:
        pa.terminate()
        raise RuntimeError(
            "No audio input device found.\n"
            "Check that your microphone is connected and recognized by the OS."
        )
    return pa


def stream_chunks():
    """Yield 30ms raw PCM chunks from the default microphone indefinitely."""
    audio = _open_pyaudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    try:
        while True:
            yield stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def record_until_silence(silence_ms: int = 1500, vad_aggressiveness: int = 2) -> bytes:
    """Record from the mic until silence is detected; return WAV bytes.

    Waits for speech to begin, then stops after silence_ms of continuous silence.
    Returns empty bytes if no speech is detected within ~10 seconds.
    """
    vad = webrtcvad.Vad(vad_aggressiveness)
    audio = _open_pyaudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    silence_limit = silence_ms // CHUNK_MS
    no_speech_limit = 10_000 // CHUNK_MS  # 10s timeout waiting for speech

    frames = []
    silence_count = 0
    no_speech_count = 0
    recording = False

    print("Listening... (speak to start)")
    try:
        while True:
            data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            is_speech = vad.is_speech(data, SAMPLE_RATE)

            if is_speech:
                if not recording:
                    print("Recording...")
                    recording = True
                silence_count = 0
                frames.append(data)
            elif recording:
                frames.append(data)
                silence_count += 1
                if silence_count >= silence_limit:
                    print("Done.")
                    break
            else:
                no_speech_count += 1
                if no_speech_count >= no_speech_limit:
                    print("(timeout — no speech detected)")
                    break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    if not frames:
        return b""

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    return buf.read()
