# Translation Agent — Plan & Roadmap

## Vision

Replace a manual workflow (watch video → hand-transcribe Japanese → hand-translate) with a semi-autonomous pipeline that produces time-aligned JA/EN output, flags uncertain segments, and keeps a human in the loop for review.

Long-term: evolve from a deterministic pipeline into a multi-agent system with critic/repair loops, glossary memory, and style modes — not one-shot LLM calls.

## Original workflow → New workflow

| Before | After (today) |
|--------|---------------|
| Manually watch and transcribe | Paste URL → auto transcribe (Whisper + captions) |
| Manually translate | Ollama translates in paragraph windows |
| No timestamps | Time-aligned segments + SRT exports |
| No quality signals | QA flags per segment |
| All-or-nothing | Two-phase: review transcript before translating |

## Phase 1 — MVP ✅ (implemented)

Deterministic pipeline with human-in-the-loop review between transcription and translation.

### Pipeline stages

1. **Download** — yt-dlp extracts best audio as WAV (`core/downloader.py`)
2. **Transcribe** — Japanese captions via yt-dlp + Whisper (mlx on Apple Silicon or faster-whisper on CPU); hybrid merge (`core/captions.py`, `core/transcriber.py`, `core/merger.py`)
3. **Segment** — sentence splitting, paragraph windows for translation (`core/segmenter.py`)
4. **Review pause** — job stops at `transcribed`; user edits `segments.json` or uses Streamlit editor
5. **Translate** — Ollama (Qwen2.5) in paragraph-window batches with JSON response parsing (`agents/translator.py`)
6. **QA** — heuristic flags: `low_confidence`, `length_anomaly`, `empty_translation`, `very_short` (`core/qa.py`)
7. **Output** — `output.txt`, `output.json`, `output.ja.srt`, `output.en.srt` (`core/output.py`)

### Infrastructure

| Component | File(s) | Status |
|-----------|---------|--------|
| Job state (JSON) | `state/models.py`, `state/store.py` | ✅ |
| Artifact cache (audio, whisper, translations) | `core/cache.py` | ✅ |
| Provider abstraction | `core/providers/transcription.py`, `translation.py` | ✅ |
| Orchestrator (transcribe / translate phases) | `pipeline/orchestrator.py` | ✅ |
| CLI (`run`, `transcribe`, `translate`, `status`, `watch`, `list`) | `pipeline/cli.py` | ✅ |
| Progress bar + verbose logging | `pipeline/progress.py` | ✅ |
| Streamlit two-phase UI + segment editor | `app/streamlit_app.py` | ✅ |
| FastAPI job API | `app/main.py` | ✅ (basic) |
| Error logging | `data/jobs/<id>/error.log` | ✅ |
| Smoke test | `scripts/smoke_test.py` | ✅ |

### Key design decisions (made during build)

| Decision | Rationale |
|----------|-----------|
| mlx-whisper over faster-whisper on M1 | CPU-only whisper hung for 16+ hours on 4.5h audio; mlx uses GPU (~realtime) |
| Two-phase workflow | User needs to trim duplicates and fix transcription errors before expensive translation |
| Per-window translation cache | ~500 Ollama calls per long video; cache enables resume and honors transcript edits |
| `segments.json` as review artifact | Simple, editable JSON; translate phase reads it back |
| Paragraph windows (≤8 sentences, ≤30s) | Balance context vs. model dropping lines in long batches |
| Local-first (Ollama + mlx) | Free; OpenAI backends stubbed for optional future use |

### Known limitations

- **Translation is slow** — ~500 Ollama calls for a 4h video (~1–2 hours). Mitigations: `qwen2.5:7b`, bigger windows, or translation cache (implemented).
- **`length_anomaly` false positives** — Japanese is compact; English is often >2.5× longer even when correct.
- **No speaker diarization** — all speech treated as one stream.
- **No visual context** — audio-only; can't resolve references to slides/images.
- **API doesn't expose two-phase endpoints yet** — CLI and Streamlit do; FastAPI still runs full pipeline.
- **Critic/repair wired** — LangGraph loop after translation (`agents/refinement.py`)

---

## Phase 2 — Agentic refinement loop ✅ (implemented)

Iterative quality improvement after baseline translation via LangGraph.

### Agents

| Agent | Role | Status |
|-------|------|--------|
| **Translator** | JA→EN in paragraph windows | ✅ Done |
| **Critic** | Compare JA/EN; score confidence, flag issues, suggest fixes | ✅ Done |
| **Repair** | Re-translate flagged segments with critic feedback | ✅ Done |

Loop (`agents/refinement.py`): `critique → repair → re-critique` until confidence threshold met or `REFINEMENT_MAX_ITERATIONS` reached.

Disable with `REFINEMENT_ENABLED=false` or `--no-refinement` on CLI.

### Still planned (Phase 2+)

- LangGraph state persistence across jobs
- FastAPI separate transcribe/translate/refine endpoints
- Evaluation metrics dashboard (edit distance, correction rate)

---

## Phase 3 — Quality & personalization ⏳ (planned)

| Feature | Description |
|---------|-------------|
| Glossary memory | JSON store of technical terms / proper nouns injected into translation prompts |
| Translation style modes | Literal (legal), natural (subtitles), summary (meetings) |
| Speaker-aware translation | Basic diarization → per-speaker context |
| Smarter QA | Replace length heuristic with critic-based scoring |
| Sub-progress during long steps | Finer-grained progress for Whisper/Ollama within a stage |

---

## Phase 4 — Distribution ⏳ (planned)

| Item | Status |
|------|--------|
| GitHub repo + README | In progress |
| Architecture diagram | TODO |
| `/examples` with real input/output samples | TODO |
| Demo video | TODO |
| HuggingFace Spaces / hosted UI | TODO |
| OpenAI backend implementation | Stubbed |

---

## Challenges (ongoing)

- Japanese speech is highly contextual; visual references (slides, gestures) are lost in audio-only transcription.
- Long videos (4+ hours) stress every stage — caching and two-phase review are essential.
- Whisper segments can be noisy (duplicates, hallucinations on silence) — manual review step addresses this.
- Subtitle-length English from short Japanese clauses triggers QA false positives.

## Config reference

See [README.md](README.md) for full `.env` options. Key knobs:

```
TRANSCRIPTION_BACKEND=mlx          # Apple Silicon GPU (fast)
TRANSLATION_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:14b
USE_ARTIFACT_CACHE=true
WHISPER_MODE=always
```
