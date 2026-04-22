"""Environment check — run this before batch_mode.py / realtime_mode.py.

Usage:
  uv run python check.py
"""

import os
import sys

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-ja-0.22")
OK = "\033[32m[OK]\033[0m"
FAIL = "\033[31m[NG]\033[0m"
WARN = "\033[33m[--]\033[0m"


def check(label: str, fn):
    try:
        msg = fn()
        print(f"  {OK}  {label}" + (f" — {msg}" if msg else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False


print("\n=== sasayakey environment check ===\n")

failures = 0

# --- imports ---
print("[1] Library imports")
for name in ("pyaudio", "webrtcvad", "vosk", "faster_whisper", "anthropic"):
    if not check(f"import {name}", lambda n=name: __import__(n) and None):
        failures += 1

# --- audio device ---
print("\n[2] Audio input device")
def _check_mic():
    import pyaudio, os, contextlib
    @contextlib.contextmanager
    def _quiet():
        devnull = os.open(os.devnull, os.O_WRONLY)
        old = os.dup(2); os.dup2(devnull, 2); os.close(devnull)
        try: yield
        finally: os.dup2(old, 2); os.close(old)
    with _quiet():
        pa = pyaudio.PyAudio()
    devs = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0]
    pa.terminate()
    if not devs:
        raise RuntimeError("no input device found — plug in a microphone")
    return f"{len(devs)} device(s): {devs[0]['name']}"

if not check("microphone", _check_mic):
    failures += 1

# --- Vosk model ---
print("\n[3] Vosk model")
def _check_vosk():
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(
            f"model not found at '{VOSK_MODEL_PATH}'\n"
            "     Run: wget https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip && unzip vosk-model-small-ja-0.22.zip"
        )
    return VOSK_MODEL_PATH

if not check(f"Vosk model ({VOSK_MODEL_PATH})", _check_vosk):
    failures += 1

# --- faster-whisper model ---
print("\n[4] faster-whisper model")
def _check_whisper():
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dirs = [d for d in os.listdir(cache) if "faster-whisper" in d] if os.path.exists(cache) else []
    if not model_dirs:
        raise RuntimeError(
            "model not cached yet — will auto-download (~465 MB) on first run of batch_mode.py"
        )
    return f"cached: {model_dirs[0]}"

r = check("faster-whisper small (cached)", _check_whisper)
if not r:
    print(f"  {WARN}  Will auto-download on first run (needs internet)")

# --- API key ---
print("\n[5] Anthropic API key")
def _check_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — export it or add to .env")
    return f"set (sk-...{key[-4:]})"

if not check("ANTHROPIC_API_KEY", _check_key):
    failures += 1

# --- webrtcvad functional ---
print("\n[6] webrtcvad VAD smoke test")
def _check_vad():
    import webrtcvad, struct, math
    vad = webrtcvad.Vad(2)
    N = 480
    silence = b"\x00" * (N * 2)
    tone = struct.pack("<" + "h" * N, *[int(30000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(N)])
    assert not vad.is_speech(silence, 16000), "silence flagged as speech"
    assert vad.is_speech(tone, 16000), "tone not flagged as speech"

check("VAD (silence→False, tone→True)", _check_vad)

# --- summary ---
print()
if failures == 0:
    print("All checks passed. You're good to go!\n")
    print("  uv run python realtime_mode.py   # real-time (Vosk)")
    print("  uv run python batch_mode.py      # batch (Whisper + Claude)\n")
else:
    print(f"{failures} check(s) failed. Fix the issues above before running.\n")
    sys.exit(1)
