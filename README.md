# Translation Agent

Japanese → English transcription and translation pipeline for long-form video (YouTube, Vimeo). Local-first: **mlx-whisper** on Apple Silicon for transcription, **Ollama (Qwen2.5)** for translation.

Paste a video URL, get time-aligned Japanese transcription, review/edit it, then translate to English with QA flags and subtitle exports.

## What it does today

| Stage | Status | Notes |
|-------|--------|-------|
| Download audio | ✅ | yt-dlp, `bestaudio`, WAV output |
| Japanese captions + Whisper | ✅ | Hybrid merge; mlx GPU on Apple Silicon |
| Segmentation | ✅ | Sentence boundaries + paragraph windows |
| **Manual review pause** | ✅ | Edit `segments.json` before translating |
| Translation | ✅ | Ollama, paragraph-window batches, resumable cache |
| **Critic / Repair loop** | ✅ | LangGraph: critique → repair → re-check (configurable) |
| QA flags | ✅ | Heuristic + critic scores (`low_translation_confidence`, etc.) |
| Outputs | ✅ | `output.txt`, `output.json`, `.srt` files |
| CLI (run / transcribe / translate) | ✅ | Progress bar + verbose mode |
| Streamlit UI | ✅ | Two-phase buttons + in-browser editor |
| FastAPI | ✅ | Basic job API |

## Pipeline

```
Video URL
  → download audio (cached)
  → fetch JA captions + Whisper transcribe (cached)
  → merge + segment
  → [PAUSE] status = transcribed — review segments.json
  → translate windows via Ollama (cached per window)
  → critic/repair loop (LangGraph, optional)
  → QA flags
  → output.txt / output.json / output.ja.srt / output.en.srt
```

Job statuses: `pending` → `downloading` → `transcribing` → `segmenting` → **`transcribed`** (review) → `translating` → **`refining`** → `completed` | `failed`

## Setup

**Prerequisites:** macOS with Apple Silicon (for mlx backend), or any OS with CPU (use `local` backend).

```bash
brew install ffmpeg yt-dlp ollama
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # adjust models if needed
ollama serve                  # separate terminal
ollama pull qwen2.5:14b       # or qwen2.5:7b on 8GB RAM
python scripts/smoke_test.py  # verify config + Ollama
```

## Running jobs

### One-shot (both phases, no review pause)

```bash
python -m pipeline.cli run "https://vimeo.com/VIDEO_ID"
python -m pipeline.cli run -v "URL"          # stage logs instead of progress bar
python -m pipeline.cli run --no-progress "URL"
```

### Two-phase (recommended for long videos)

```bash
# Phase 1 — download + transcribe + segment, then STOP
python -m pipeline.cli transcribe "URL"
# → writes data/jobs/<job_id>/segments.json, status = transcribed

# Review: edit segments.json — delete duplicates, fix japanese text

# Phase 2 — translate reviewed segments + critic/repair + write outputs
python -m pipeline.cli translate <job_id>
python -m pipeline.cli translate <job_id> --no-refinement   # skip critic/repair (faster)
```

Re-run an existing job:

```bash
python -m pipeline.cli transcribe --job-id <job_id> -v
python -m pipeline.cli translate <job_id> -v
```

### Other CLI commands

```bash
python -m pipeline.cli list
python -m pipeline.cli status <job_id>
python -m pipeline.cli watch <job_id>        # follow a job started elsewhere
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

- **Transcribe** — phase 1 only, then pause for review
- **Run all** — both phases end-to-end
- In-browser segment editor: edit Japanese, delete rows, save, then **Translate**

### API

```bash
uvicorn app.main:app --reload
# POST /jobs  {"youtube_url": "..."}
# GET  /jobs/{job_id}
```

## Outputs

Written to `data/jobs/<job_id>/`:

| File | When | Purpose |
|------|------|---------|
| `segments.json` | After transcribe | Editable Japanese transcript (input to translate phase) |
| `output.txt` | After translate | Human-readable JA/EN blocks — **final result for reading** |
| `output.json` | After translate | Machine-readable segments + metadata |
| `output.ja.srt` | After translate | Japanese subtitles (video players, editors) |
| `output.en.srt` | After translate | English subtitles |
| `refinement_log.json` | After translate (if refinement on) | Critic/repair stats |
| `error.log` | On failure | Full Python traceback |
| `whisper_raw.json` | After transcribe | Raw Whisper output |
| `audio.wav` | After download | Extracted audio |

### QA flags (in `output.txt` / `output.json`)

| Flag | Meaning |
|------|---------|
| `low_confidence` | Whisper transcription confidence < 0.7 |
| `low_translation_confidence` | Critic translation score < 0.7 |
| `critic_flagged` | Critic flagged for repair |
| `critic_repaired` | Repair agent updated this segment |
| `length_anomaly` | English >2.5× longer than Japanese (heuristic) |
| `empty_translation` | English field is blank |
| `very_short` | Japanese text under 2 characters |

Flags are review hints, not automatic errors. `length_anomaly` fires often on normal JA→EN because English is typically longer.

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIPTION_BACKEND` | `mlx` | `mlx` (Apple Silicon GPU) \| `local` (faster-whisper CPU) \| `openai` |
| `MLX_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | HuggingFace repo for mlx backend |
| `LOCAL_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model when `backend=local` |
| `TRANSLATION_BACKEND` | `ollama` | `ollama` \| `openai` |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Translation model (`qwen2.5:7b` for 8GB RAM) |
| `WHISPER_MODE` | `always` | `always` \| `fallback_only` (skip Whisper if captions cover ≥95%) |
| `USE_ARTIFACT_CACHE` | `true` | Reuse audio/captions/whisper/translations per URL |
| `REFINEMENT_ENABLED` | `true` | Run critic/repair loop after translation |
| `REFINEMENT_CONFIDENCE_THRESHOLD` | `0.7` | Re-translate segments with critic score below this |
| `REFINEMENT_MAX_ITERATIONS` | `2` | Max critique→repair cycles |
| `YTDLP_COOKIES_FROM_BROWSER` | — | For login-required Vimeo/YouTube (`chrome`, `safari`, etc.) |

## Artifact cache

Re-running the same URL reuses work from `data/cache/<url_hash>/`:

| Cached artifact | Skips |
|-----------------|-------|
| `audio.wav` | Re-download |
| `captions.ja.vtt` / `.srt` | Re-fetch captions |
| `whisper_raw.json` | Re-transcribe (invalidated if whisper model changes) |
| `translations.json` | Re-translate unchanged windows |

Disable with `USE_ARTIFACT_CACHE=false`.

## Transcription backends

- **`mlx`** (default, Apple Silicon) — GPU via `mlx-whisper`. ~realtime or faster. First run downloads model weights (~1.6GB).
- **`local`** — `faster-whisper` on CPU. Portable but very slow on long audio. Uses VAD + all CPU cores.
- **`openai`** — not implemented.

## Project layout

```
translation-agent/
├── agents/           # translator, critic, repair, refinement (LangGraph)
├── app/              # FastAPI + Streamlit
├── core/             # downloader, captions, transcriber, merger, segmenter, qa, output, cache
├── pipeline/         # orchestrator, cli, progress
├── state/            # Job/Segment models, JSON persistence
├── scripts/          # smoke_test.py
├── data/
│   ├── jobs/<id>/    # per-job artifacts
│   └── cache/<hash>/ # shared cache per video URL
├── config.py
├── .env.example
└── PLAN.md           # roadmap and architecture notes
```

## Roadmap

See [PLAN.md](PLAN.md) for what's planned next (critic/repair loop, glossary, diarization, etc.).
