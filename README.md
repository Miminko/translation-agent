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
- Speech to Japanese
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
- Qwen (Ollama) + faster-whisper (local, free)
- yt-dlp (youtube download)
- ffmpeg (audio extraction)
- LangGraph (agent framework, phase 2)
- Streamlit + FastAPI

MVP: Deterministic Pipeline
1. Download video + extract audio
```bash
yt-dlp -x --audio-format mp3 <youtube_url>
```

2. Transcribe Japanese

3. Segment normalization
Group meaningful chunks (sentence boundaries).

4. Translation (baseline)

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
1. `brew install ffmpeg yt-dlp ollama`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp .env.example .env` and adjust model names if needed
5. `ollama serve` (separate terminal)
6. `ollama pull qwen2.5:14b` (or `qwen2.5:7b` on 8GB RAM)
7. `python scripts/smoke_test.py` — verifies config + Ollama translation
8. `python scripts/smoke_test.py --audio path/to/audio.wav` — also tests Whisper

RUNNING JOBS:
```bash
# One-shot: process a video URL end-to-end (progress bar on by default)
python -m pipeline.cli run "https://www.youtube.com/watch?v=VIDEO_ID"
python -m pipeline.cli run -v "URL"          # stage logs instead of progress bar
python -m pipeline.cli run --no-progress "URL"

python -m pipeline.cli watch <job_id>        # follow a job started elsewhere
python -m pipeline.cli list
python -m pipeline.cli status <job_id>

# API
uvicorn app.main:app --reload
# POST /jobs  {"youtube_url": "..."}
# GET  /jobs/{job_id}

# Streamlit review UI (two-phase buttons + in-browser transcript editor)
streamlit run app/streamlit_app.py
```

TWO-PHASE WORKFLOW (review transcription before translating):
```bash
# Phase 1 — download + transcribe + segment, then STOP for review
python -m pipeline.cli transcribe "URL"
# -> writes data/jobs/<job_id>/segments.json and leaves status = "transcribed"

# Review/edit the transcript: open segments.json and
#   - delete entries to drop duplicates / junk
#   - fix any `japanese` text errors
# (edits are honored on translate; only changed windows are re-translated)

# Phase 2 — translate the reviewed segments and write outputs
python -m pipeline.cli translate <job_id>
```
Translation is resumable and cached per window in `data/cache/<url>/translations.json`,
so an interrupted `translate` run picks up where it left off, and unchanged
transcript regions are reused after edits.

Outputs are written to `data/jobs/<job_id>/`:

- `segments.json` — editable transcript for review (input to the translate phase)
- `output.txt` — aligned JA/EN blocks
- `output.json` — machine-readable segments + flags
- `output.ja.srt` / `output.en.srt` — subtitles
- `error.log` — full traceback if a phase fails

**Artifact cache:** Re-running the same video URL reuses cached audio, captions, Whisper output, and per-window translations from `data/cache/` (skips download + transcribe, and resumes/reuses translation). Disable with `USE_ARTIFACT_CACHE=false` in `.env`.

**Transcription backend (`TRANSCRIPTION_BACKEND`):**

- `mlx` (default, **Apple Silicon only**) — runs Whisper on the M-series GPU via `mlx-whisper`. Near-realtime; a 1-hour video transcribes in minutes. Model set by `MLX_WHISPER_MODEL` (default `mlx-community/whisper-large-v3-turbo`).
- `local` — `faster-whisper` on CPU. Portable but slow for `large-v3` on long audio (many hours). Uses VAD to skip silence and all CPU cores. Model set by `LOCAL_WHISPER_MODEL` (`small`/`medium`/`large-v3-turbo`/`large-v3`).
- `openai` — not implemented yet.

Whisper progress is printed live to stderr (`whisper NN% (mm:ss / mm:ss)`) so a slow run is distinguishable from a hung one.

SETTING UP & TESTING LOCAL MODELS:
- I've opted for free local models since I'm cheap.
- Run `ollama serve` to spin up server in a separate terminal
- Run `ollama pull qwen2.5:14b` to download 14 billion param Qwen model
- Run `ollama run qwen2.5:14b` to test model within terminal
- Run curl command to API to test HTTP requests:
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:14b",
  "messages": [{"role": "user", "content": "Translate to English: こんにちは、元気ですか？"}],
  "stream": false
}'
```
- To test local Whisper model:
```bash
python -c "
from faster_whisper import WhisperModel
print('Downloading model on first run...')
m = WhisperModel('small', device='cpu', compute_type='int8')
print('Ready:', m)
"
```


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
