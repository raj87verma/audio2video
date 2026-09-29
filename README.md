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
3b. Content-theme detect  (filename + lyrics text)  → overrides generic mood
       │                    keywords when a devotional/spiritual theme is
       │                    detected (see "Content-aware visuals" below)
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

### Content-aware visuals

The acoustic mood classifier (step 2) only looks at tempo/loudness/timbre
— it has no idea what a song is actually *about*, which can pick oddly
generic keywords (e.g. "celebration", "dance") for devotional music that
happens to be acoustically upbeat. To help with this, the pipeline also
checks the audio file's **filename** (and any transcribed lyrics) for
devotional/spiritual vocabulary (aarti, bhajan, kirtan, temple, and deity
names, in both English transliteration and Devanagari) — if found, shot
keywords switch to devotional-themed terms (temple, diya lamp, incense,
prayer, ...) instead of the generic mood keywords, and shots are made
longer/slower to match typical devotional pacing. Naming your file
descriptively (e.g. keeping "Aarti" or a deity's name in the filename, as
most downloaded devotional tracks already do) is what triggers this — it
requires no configuration.

Every stage degrades gracefully: with **zero API keys configured**, the app
still produces a complete video using animated procedural backgrounds and
synthetic sound effects. Configuring the free API keys below just adds
real stock footage/photos and, optionally, real recorded sound effects.

---

## Windows Installer (easiest way to get started)

If you're on Windows and don't want to set up Python yourself, download the
ready-made installer instead of following the manual setup below:

1. Go to the [Releases page](https://github.com/raj87verma/audio2video/releases)
   and download the latest `Audio2Video-Setup.exe`.
   - No release yet, or want the newest in-progress build? Go to
     [Actions → Build Windows Installer](https://github.com/raj87verma/audio2video/actions/workflows/build-windows-installer.yml),
     open the most recent successful run, and download the
     `Audio2Video-Setup-*` artifact from the bottom of the run's summary
     page (artifacts require being logged into GitHub; they expire after
     30 days — a tagged release is permanent).
2. Run `Audio2Video-Setup.exe`. It does **not** require Administrator
   rights and installs by default under your own user profile
   (`%LOCALAPPDATA%\Programs\Audio2Video`) — you can change the install
   folder to any drive/path in the setup wizard.
3. Launch **Audio2Video** from the Start Menu (or the optional desktop
   shortcut). Everything — Python, FFmpeg, and every dependency — is
   already bundled inside; nothing else to install.
4. (Optional) Add free Pexels/Pixabay/Freesound API keys in the app's
   **Settings** tab — see [section 2](#2-getting-free-api-keys-all-optional)
   below.

The installer is built automatically by
[`.github/workflows/build-windows-installer.yml`](.github/workflows/build-windows-installer.yml)
using PyInstaller (freezes the app + all Python dependencies) and
[Inno Setup](https://jrsoftware.org/isinfo.php) (wraps that into a
single-file `Setup.exe` with Start Menu/Desktop shortcuts and a proper
uninstaller). To trigger a fresh build yourself: go to
**Actions → Build Windows Installer → Run workflow**.

> **Note:** the installer is not code-signed (that requires a paid code
> signing certificate). Windows SmartScreen may show an "unrecognized app"
> warning the first time you run it — click **More info → Run anyway** to
> proceed. This is expected for any unsigned installer, not a sign that
> something is wrong.

### Building the installer locally (advanced / maintainers)

You need a real Windows machine for this (PyInstaller cannot cross-compile
a Windows `.exe` from Linux/macOS):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller packaging\audio2video.spec --noconfirm
```

Then install [Inno Setup](https://jrsoftware.org/isdl.php), and run:

```powershell
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=0.1.0 packaging\installer.iss
```

The finished installer is written to `dist\installer\Audio2Video-Setup.exe`.

---

## 1. Setup (from source, any OS)

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
├── run.py                   PyInstaller / standalone entry point (python run.py)
├── app/
│   ├── main.py              GUI entry point (python -m app.main)
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
├── packaging/
│   ├── app_icon.ico          app/installer icon
│   ├── audio2video.spec      PyInstaller build spec
│   └── installer.iss         Inno Setup script (builds Setup.exe)
├── .github/workflows/
│   └── build-windows-installer.yml  CI: builds Setup.exe on windows-latest
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
  more expensive than still-image Ken Burns animation. As a rough guide,
  a ~6-minute track at 1080p typically renders in the range of tens of
  minutes rather than hours; very long tracks (10+ minutes) or 4K output
  will take proportionally longer. Lowering the resolution/fps in Settings
  is the most effective way to speed up a render.
- Content-theme detection (see above) currently only recognizes a curated
  devotional/spiritual vocabulary — other specific themes (e.g. weddings,
  travel vlogs) still rely on the generic acoustic mood classifier's
  keywords.
