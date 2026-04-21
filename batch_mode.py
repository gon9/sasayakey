"""Pattern 1: Record → transcribe → send to Claude.

Flow:
  Press Enter → speak → silence detected → faster-whisper transcription
  → Claude API → display reply → repeat

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python batch_mode.py
"""

import anthropic

from utils.asr import transcribe_batch
from utils.audio import record_until_silence


def main() -> None:
    client = anthropic.Anthropic()
    history: list[dict] = []

    print("=== Batch Voice AI ===")
    print("Press Enter to record, Ctrl-C to quit.\n")

    while True:
        try:
            input("[ Enter to record ]")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        wav = record_until_silence()
        if not wav:
            continue

        print("Transcribing...")
        text = transcribe_batch(wav)
        if not text:
            print("(no speech recognized)\n")
            continue

        print(f"\nYou: {text}")

        history.append({"role": "user", "content": text})
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=history,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        print(f"Claude: {reply}\n")


if __name__ == "__main__":
    main()
