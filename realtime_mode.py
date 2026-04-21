"""Pattern 2: Real-time transcription with Vosk (free, local, streaming).

Architecture:
  mic → webrtcvad (30ms chunks) → Vosk KaldiRecognizer (true streaming)
    → display partial text in place
    → on 1.5s silence: flush utterance → Claude API → display reply

How real-time ASR works here:
  - webrtcvad checks each 30ms chunk for voice activity (fast signal processing)
  - Vosk processes every chunk and emits partial results immediately
  - No waiting for silence to start transcription — text appears as you speak
  - Silence detection is only used to decide "utterance is done → send to LLM"

Setup:
  wget https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip
  unzip vosk-model-small-ja-0.22.zip
  export ANTHROPIC_API_KEY=sk-...
  export VOSK_MODEL_PATH=vosk-model-small-ja-0.22  # default if unset

Usage:
  python realtime_mode.py
  python realtime_mode.py --no-llm   # transcription only, skip Claude
"""

import os
import sys

import anthropic
import webrtcvad

from utils.asr import VoskRealtimeASR
from utils.audio import CHUNK_MS, SAMPLE_RATE, stream_chunks

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-ja-0.22")
SILENCE_MS = 1500  # ms of silence to mark end of utterance
SILENCE_LIMIT = SILENCE_MS // CHUNK_MS


def main(use_llm: bool = True) -> None:
    try:
        asr = VoskRealtimeASR(model_path=VOSK_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    client = anthropic.Anthropic() if use_llm else None
    history: list[dict] = []
    vad = webrtcvad.Vad(2)

    print("=== Realtime Voice AI ===")
    if not use_llm:
        print("(LLM disabled — transcription only)")
    print("Speak naturally. Ctrl-C to quit.\n")

    silence_count = 0
    confirmed_segments: list[str] = []
    in_speech = False

    try:
        for chunk in stream_chunks():
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            text, is_final = asr.process(chunk)

            if is_speech:
                silence_count = 0
                in_speech = True
                # Collect internally-finalized segments
                if is_final and text:
                    confirmed_segments.append(text)
                # Show live partial text in place
                display = " ".join(confirmed_segments + ([text] if text else []))
                if display:
                    print(f"\r\033[K[...] {display}", end="", flush=True)

            elif in_speech:
                silence_count += 1
                if silence_count >= SILENCE_LIMIT:
                    # Flush whatever remains in Vosk's buffer
                    tail = asr.flush()
                    if tail:
                        confirmed_segments.append(tail)

                    full_text = " ".join(confirmed_segments).strip()
                    confirmed_segments = []
                    in_speech = False
                    silence_count = 0

                    if not full_text:
                        continue

                    print(f"\r\033[KYou: {full_text}")

                    if client is not None:
                        history.append({"role": "user", "content": full_text})
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=512,
                            messages=history,
                        )
                        reply = response.content[0].text
                        history.append({"role": "assistant", "content": reply})
                        print(f"Claude: {reply}\n")

    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    main(use_llm=not no_llm)
