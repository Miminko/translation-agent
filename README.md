Semi-autonomous translation pipeline agent that helps me complete Japanese-English transcription and translation jobs.

Current Workflow:
- Manually watch videos and manually transcribe Japanese speech
- Manually translate transcription to English

New Workflow:
- Videos -> audio -> Japanese Transcription -> segmentation -> English Translation -> QA Checks -> Review Output
- Multiple specialized roles
- Iterative refinement loops
- Explicit evaluation steps (quality + consistency checks)
- Human-in-the-loop approval
- Paste in Youtube URL, click run, get Japanese transcript (time-aligned), English translation (aligned per segment), optional term consistency notes, confidence flags per segments
- Manually review flagged portions

Challenges:
- A lot of human speech (especially Japanese) is contextual, requires visual understanding as opposed to pure audio

High Level:
- Youtube Downloader
- Audio Extractor
- Speech to Japanese (Whisper API)
- Segmentation Agent (chunk + align timestamps)
- Translation Agent (JA -> EN)
- QA Agent (consistency checks, missing meaning detection, hallucination risk scoring)
- Review Interface (me)
- Multi-step reasoning loop, not just one-shot LLM calls
- State Tracking: store segments, revisions, critic feedback history
- Conditional execution: e.g. if confidence < 0.7: rerun_translation()
- Tool Use: agent should decide whether to re-transcribe noisy segments, expand context window, fetch prior/subsequent sentences
- Evaluation Metrics: track translation edit distance after critique, confidence distributions, correction rate
- Human-in-the-loop UI: highlight uncertain segments, suggested fixes, approve/reject workflow

- Extra upgrades: Add glossary memory by building a simple store in JSON for technical terms or jargon, that get injected into translation prompts ==> persistent agent system.
- Add speaker-aware translation, basic diarization
- Add translation style modes: e.g. literal mode (for legal documents), natural mode (subtitles), summary mode (for meetings)

Tech Stack:
- Python
- OpenAI API (or Whisper + GPT-4.1 / GPT-5-mini class model)
- yt-dlp (youtube download)
- ffmpeg (audio extraction)
- LangGraph (agent framework)
- React + FastAPI backend

MVP: Deterministic Pipeline
1. Download video + extract audio
```bash
yt-dlp -x --audio-format mp3 <youtube_url>
```

2. Transcribe Japanese
```python
from openai import OpenAI
client = OpenAI()

audio_file = open("audio.mp3", "rb")

transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="ja",
    response_format="verbose_json"
)
```

3. Segment normalization
Group meaningful chunks (sentence boundaries).

4. Translation (baseline)
For each segment:
```python
def translate_segment(text):
    return client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
Translate Japanese → natural English.

Rules:
- preserve meaning, not literal structure
- keep tone (formal/informal)
- preserve names
- do NOT add explanations

Text:
{text}
"""
    )
```

5. Output aligned file
Generate as Google Doc format:
start_time:
end_time:
japanese:
english:

6. Convert to Agentic System
- 3 agents: Translation Agent, Critic Agent, Repair Agent
- Translation Agent: detects ambiguity, chooses literal vs adaptive translation, preserve discourse flow
- Critic Agent: checks whether English preserves meaning, if anything imitted, tone match, hallucinations
```
You are a tranlation critic.
Compare Japanese and English.
Return:
- issues (if any)
- confidence 0-1
- corrected translation if needed
```
- Repair Agent: if critic flags issues -> regenerate translation with feedback.
- Creates loop: translate -> critique -> re-check


SETUP INSTRUCTIONS:
1. `brew install ffmpeg yt-dlp`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`


TODO:
- upload to github repo
- generate architecture diagram
- setup intructions to run locally
- example input/outputs
- explanation of how it works
- /examples folder with real transcripts and translations
- logs of agent decisions
- record demo video
- host HuggingFace spaces, vercel SPA
