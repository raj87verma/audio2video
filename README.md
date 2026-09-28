# Audio2Video

Turn an audio file (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, ...) into an
automatically edited, beat-synced video — using **only free and open
resources**. No paid APIs, no subscriptions required.

> **What this is:** a rule-based, signal-driven "AI music video maker" that
> analyzes your audio's tempo/energy/mood, cuts a beat-synced timeline,
> fetches matching free stock photos/videos (or generates animated
> procedural visuals when it can't), and layers auto-generated sound
> effects on top — fully offline-capable if you skip the optional API keys.
>
> **What this is *not*:** a text-to-video AI model that generates
> photorealistic novel footage from scratch. All real-world footage comes
> from free stock media libraries (Pexels/Pixabay); anything else is
> procedurally generated (animated gradients/particles/waveforms), not
> AI-hallucinated video.

---

## How it works

```
   audio file
       │
       ▼
1. Audio analysis        (librosa)        → tempo, beats, energy, spectral features
       │
       ▼
2. Mood classification    (rule-based)     → mood label, keywords, color grade, cut speed
       │
       ▼
3. Lyrics/speech detect   (faster-whisper, optional, local) → keywords from vocals
       │
       ▼
4. Shot planning                          → beat-aligned list of shots
       │
       ▼
5. Stock media fetch      (Pexels / Pixabay, optional)  → photo/video per shot
       │                  ↳ falls back to generated visuals if no key / no match
       ▼
6. Video assembly         (MoviePy / FFmpeg)
       ↳ Ken Burns zoom/pan on stills, crossfades, mood color grading
       ↳ synthetic (+ optional Freesound) SFX layered under your audio
       ▼
   finished .mp4
```

Every stage degrades gracefully: with **zero API keys configured**, the app
still produces a complete video using animated procedural backgrounds and
synthetic sound effects. Configuring the free API keys below just adds
real stock footage/photos and, optionally, real recorded sound effects.

---

## 1. Setup

### Requirements
- Python 3.10+ (a virtual environment is strongly recommended)
- FFmpeg (system-installed, or rely on the bundled `imageio-ffmpeg` binary)

### Install

```bash
cd audio2video
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If your system doesn't have `ffmpeg` on PATH, MoviePy will automatically
fall back to the `imageio-ffmpeg` bundled binary that pip installed for
you — no extra step needed for basic use. If you want a full-featured
system FFmpeg (recommended for best compatibility), install it via your
package manager (e.g. `apt install ffmpeg`, `brew install ffmpeg`) or grab
a static build from https://johnvansickle.com/ffmpeg/.

### Run

```bash
python -m app.main
```

This opens the desktop GUI with three tabs: **Create Video**, **Settings**,
and **About / Help**.

---

## 2. Getting free API keys (all optional)

The app works with **none** of these configured. Adding them improves the
variety/realism of the visuals and sound effects.

| Provider | Used for | Free tier | Sign-up |
|---|---|---|---|
| **Pexels** | stock photos & videos | Free, generous rate limits, no credit card | https://www.pexels.com/api/ |
| **Pixabay** | stock photos & videos | Free, generous rate limits, no credit card | https://pixabay.com/api/docs/ |
| **Freesound** | real keyword-matched sound effects | Free, no credit card | https://freesound.org/apiv2/apply/ |

Steps (same idea for all three):
1. Create a free account on the provider's site.
2. Find their "API" / "Developers" page (linked above) and request/generate
   an API key or token.
3. Open the app → **Settings** tab → paste the key into the matching field
   → **Save Settings**.

Settings are stored locally at `~/.audio2video/settings.json` (never
uploaded anywhere).

---

## 3. Using the app

1. Open the **Create Video** tab.
2. Click **Choose Audio File...** and pick an `.mp3`/`.wav`/etc.
3. (Optional) Click **Save As...** to choose a custom output location —
   otherwise a timestamped file is created under `~/.audio2video/output/`.
4. Click **Create Video**. Progress and status messages stream into the
   log box as the app analyzes the audio, plans shots, fetches visuals,
   and renders the final MP4.
5. When finished, click **Open Output Folder** to locate your video.

You can **Cancel** at any point; the pipeline stops at the next safe
checkpoint (in-progress downloads/renders are not partially corrupted).

### Settings reference

- **Resolution** — 720p / 1080p / 4k output.
- **Frame rate** — 10–60 fps.
- **Prefer stock video clips over stills** — when unchecked, the app
  prefers photos over video clips (faster to fetch/render).
- **Detect lyrics/speech** — runs a local, offline Whisper model on your
  audio to pull extra visual search keywords out of any vocals/speech.
  First use downloads a small open-weight model (one-time, then cached).
  Turn this off for faster runs on instrumental-only tracks.
- **Whisper model size** — tiny/base/small/medium. Bigger = more accurate
  transcription but slower and more memory.

---

## 4. What "SFX and VFX added automatically" actually means here

- **VFX**: Ken Burns zoom/pan on still images, mood-based color grading
  (5 built-in "grades" — cool/contrasty, warm/vibrant, moody/desaturated,
  soft pastel, neutral cinematic), crossfade transitions between shots, and
  — whenever no stock footage matches — a fully generated animated
  background (gradient + drifting particles + a live waveform visualizer
  reacting to your track's actual loudness).
- **SFX**: a layer of procedurally synthesized whooshes, impacts, risers
  and sparkles, automatically placed at shot transitions and at strong
  musical accents (onsets) in high-energy sections — mixed underneath your
  original audio at a low relative volume. If you add a free Freesound API
  key, the app can additionally pull real recorded sound effects matching
  a shot's keywords instead of only synthetic ones.

---

## 5. Licensing / usage notes

- **Pexels** and **Pixabay** content is free to use, including for
  commercial projects, under their own license terms — always check the
  current license text on their sites (linked above) before redistributing
  anything you make with this tool, especially commercially.
- **Freesound** sounds have per-upload licenses (many are Creative
  Commons); if you rely on real Freesound effects (not just the synthetic
  ones), check the individual sound's license on freesound.org.
- The **procedurally generated visuals and synthetic SFX** produced by
  this app are code-generated locally and carry no third-party licensing
  restrictions.
- This tool doesn't remove or add any attribution automatically — if a
  provider's terms require attribution for a specific use case, add it
  yourself.

---

## 6. Project structure

```
audio2video/
├── app/
│   ├── main.py              entry point (python -m app.main)
│   ├── config.py            Settings dataclass, cache/output paths
│   ├── core/
│   │   ├── audio_analysis.py    tempo/beats/energy/spectral features (librosa)
│   │   ├── mood.py              rule-based mood classifier
│   │   ├── transcribe.py        optional local speech-to-text (faster-whisper)
│   │   ├── shot_planner.py      beat-aligned shot list
│   │   ├── media_fetcher.py     Pexels/Pixabay free stock media client
│   │   ├── procedural_visuals.py fallback animated visuals (PIL/numpy)
│   │   ├── sfx.py               synthetic SFX + optional Freesound client
│   │   ├── video_builder.py     MoviePy assembly: Ken Burns, grading, mux
│   │   └── pipeline.py          orchestrates all of the above
│   └── gui/
│       ├── main_window.py       Create Video / Settings / About tabs
│       └── worker.py            background QThread pipeline runner
└── requirements.txt
```

Every module in `app/core/` can be imported and used independently of the
GUI (e.g. from a script or notebook) if you'd rather script the pipeline:

```python
from app.core.pipeline import run_pipeline
from app.config import Settings

result = run_pipeline("my_song.mp3", settings=Settings.load())
print(result.output_path, result.mood.label)
```

---

## 7. Known limitations

- This is **not** a generative-AI video model — it assembles real stock
  footage/photos and generated backgrounds, it does not invent new
  photorealistic scenes.
- Free stock media search quality depends on how well a shot's keywords
  match what's available on Pexels/Pixabay; unusual/abstract keywords may
  fall back to generated visuals more often.
- Speech-to-text keyword extraction works best on tracks with clear,
  intelligible vocals; heavily distorted/layered vocals may not transcribe
  well (the app silently falls back to mood-based keywords in that case).
- Rendering time scales with video length/resolution/fps and whether stock
  video clips (vs. stills) are used, since video decoding/re-encoding is
  more expensive than still-image Ken Burns animation.
