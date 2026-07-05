import ollama
from faster_whisper import WhisperModel


model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.wav", language="ja")

for seg in segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "Translate Japanese to natural English. Preserve tone and names."},
        {"role": "user", "content": "それでは始めましょう。"},
    ],
)
print(response["message"]["content"])
