# Audio2Video

Turn an audio file (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, ...) into an
automatically edited, beat-synced video — using **only local resources**.
No paid APIs, no online media sources, no subscriptions, nothing to sign
up for.

> **What this is:** a rule-based, signal-driven "AI music video maker" that
> analyzes your audio's tempo/energy/mood, cuts a beat-synced timeline,
> pulls matching visuals from a folder of your own photos/videos (or
> generates animated procedural visuals when none are configured), and
> layers auto-generated sound effects on top — fully offline, always.
>
> **What this is *not*:** a text-to-video AI model that generates
> photorealistic novel footage from scratch, and **not** a tool that
> fetches stock media from the internet. There is no code path anywhere
> in this app that contacts Pexels, Pixabay, Wikimedia Commons,
> Freesound, or any other online service — every real photo/video in the
> output is one you supplied yourself; anything else is procedurally
> generated locally (animated gradients/particles/waveforms), never
> downloaded or AI-hallucinated.

---

## How it works

```
   audio file
       │
       ▼
1. Audio analysis        (librosa)        → tempo, beats, energy, spectral features
       │
       ▼
2. Mood classification    (rule-based)     → mood label, color grade, cut speed
       │
       ▼
3. Lyrics/speech detect   (faster-whisper, optional, local) → used only for pacing (see below)
       │
       ▼
3b. Content-theme detect  (filename + lyrics text)  → slows shot pacing when a
       │                    devotional/spiritual theme is detected (see below)
       ▼
4. Shot planning                          → beat-aligned list of shots
       │
       ▼
5. Media resolution   (your own local media folder, round-robin across every
       │               shot; falls back to generated visuals if empty/unset)
       ▼
6. Video assembly         (MoviePy / FFmpeg)
       ↳ Ken Burns zoom/pan on stills, crossfades, mood color grading
       ↳ synthetic SFX layered under your audio
       ▼
   finished .mp4
```

### Your own media folder (the only source of real photos/videos)

Audio2Video does not fetch anything from the internet. The only way real
photos/videos end up in your output video is a folder you point it at
yourself.

**How to set it up:** open the app's **Settings** tab → find "Media
Folder" → click **Browse...** and pick any folder on your computer (new
or existing, doesn't need to be empty). Audio2Video automatically creates
two subfolders inside it:

```
<the folder you picked>/
  images/    <- put your photos here
  videos/    <- put your video clips here
```

Drop any number of files into `images/` and `videos/` (supported formats:
`.jpg`/`.jpeg`/`.png`/`.bmp`/`.webp` for images, `.mp4`/`.mov`/`.webm`/
`.mkv`/`.avi`/`.m4v` for videos). The Settings tab shows a live count of
how many of each it finds.

You don't have to use the subfolders at all — any supported image/video
file placed **directly inside the folder you picked** (not in any further
subfolder) is also detected automatically, classified by its own file
extension. This is useful if you already have a folder of photos/videos
and would rather just point Audio2Video at it than reorganize everything
into `images/`/`videos/` first.

What happens with your files:
- Every shot in every video you create cycles through the files in this
  folder — images and videos mixed together, not "all video first, then
  photos" — so a folder containing both genuinely uses the variety
  you've supplied.
- **Videos** are used in short, randomly varied 2-4 second excerpts per
  shot — never the whole clip at once. If a shot needs more than 4
  seconds (shot length varies with the song's tempo/mood), that short
  excerpt loops to fill the remaining time rather than jumping to a
  different part of the video mid-shot.
- **Photos** get the usual Ken Burns pan/zoom and mood-based color
  grading, plus a subtle animated glow + twinkling sparkle effect on
  top, so they don't look flat/static.
- If the folder is empty (or no folder is set at all), shots simply use
  an animated generated background instead — there is no online source
  to fall back to, ever.

Click **Clear** in Settings to unset the folder and go back to using only
generated visuals.

Only use media you actually have the rights to use — your own photos/
videos, or ones whose license explicitly permits this kind of use.
Audio2Video doesn't check licensing for you, since every file in that
folder is something you chose to put there yourself.

### Content-aware pacing

The acoustic mood classifier (step 2) only looks at tempo/loudness/timbre
— it has no idea what a song is actually *about*. To help with this, the
pipeline also checks the audio file's **filename** (and any transcribed
lyrics) for devotional/spiritual vocabulary (aarti, bhajan, kirtan,
temple, deity names, in both English transliteration and Devanagari) —
if found, shots are made longer/slower to match typical contemplative
devotional pacing instead of whatever cut speed the acoustic mood alone
would pick. This **only affects timing**, not which visuals are used —
visuals always come from your media folder (or generated backgrounds)
regardless of song content, since Audio2Video has no way to search for
a topic-specific visual anywhere.

Every stage degrades gracefully: with **no media folder configured**, the
app still produces a complete video using animated procedural backgrounds
and synthetic sound effects.

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
4. Open the **Settings** tab and set your media folder — see "Your own
   media folder" above.

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

## 2. Using the app

1. Open the **Settings** tab and set your media folder (see "Your own
   media folder" above) — optional, but this is the only way real
   photos/videos end up in your output.
2. Open the **Create Video** tab.
3. Click **Choose Audio File...** and pick an `.mp3`/`.wav`/etc.
4. (Optional) Click **Save As...** to choose a custom output location —
   otherwise a timestamped file is created under `~/.audio2video/output/`.
5. Click **Create Video**. Progress and status messages stream into the
   log box as the app analyzes the audio, plans shots, resolves visuals,
   and renders the final MP4.
6. When finished, click **Open Output Folder** to locate your video.

You can **Cancel** at any point; the pipeline stops at the next safe
checkpoint.

### Settings reference

- **Media Folder** — an editable path field plus **Browse...**, **Clear**,
  and **Open Folder** buttons, and a live image/video count. See "Your
  own media folder" above. Empty (the default) means every shot uses
  generated visuals.
- **Resolution** — 720p / 1080p / 4k output.
- **Frame rate** — 10–60 fps.
- **Detect lyrics/speech** — runs a local, offline Whisper model on your
  audio. Used only to help detect a devotional/spiritual theme in the
  lyrics for pacing purposes (see "Content-aware pacing" above) — it has
  no effect on which visuals are chosen. First use downloads a small
  open-weight model (one-time, then cached). Turn this off for faster
  runs on instrumental-only tracks.
- **Whisper model size** — tiny/base/small/medium. Bigger = more accurate
  transcription but slower and more memory.

Settings are stored locally at `~/.audio2video/settings.json` (never
uploaded anywhere — there's nowhere for it to be uploaded to).

---

## 3. What "SFX and VFX added automatically" actually means here

- **VFX**: Ken Burns zoom/pan on still images, mood-based color grading
  (5 built-in "grades" — cool/contrasty, warm/vibrant, moody/desaturated,
  soft pastel, neutral cinematic), crossfade transitions between shots, and
  — whenever no local media is configured or available — a fully generated
  animated background (gradient + drifting particles + a live waveform
  visualizer reacting to your track's actual loudness). Photos from your
  media folder additionally get a subtle animated glow + twinkling
  sparkle overlay.
- **SFX**: a layer of procedurally synthesized whooshes, impacts, risers
  and sparkles, automatically placed at shot transitions and at strong
  musical accents (onsets) in high-energy sections — mixed underneath your
  original audio at a low relative volume. Entirely generated locally
  with numpy; no external sound library or service involved.

---

## 4. Licensing / usage notes

- Everything in the finished video is either something you supplied
  yourself (your media folder's photos/videos) or generated locally by
  this app's own code (procedural visuals, synthetic SFX) — neither
  carries any third-party licensing restriction from Audio2Video itself.
- Only use media you actually have the rights to use in your media
  folder. This tool doesn't check licensing for you.

---

## 5. Project structure

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
│   │   ├── content_hints.py     devotional-theme detection (pacing only)
│   │   ├── shot_planner.py      beat-aligned shot list
│   │   ├── local_media.py       scans the user's own media folder
│   │   ├── procedural_visuals.py fallback animated visuals (PIL/numpy)
│   │   ├── sfx.py               synthetic SFX generator
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

settings = Settings.load()
settings.local_media_dir = "/path/to/your/media/folder"
result = run_pipeline("my_song.mp3", settings=settings)
print(result.output_path, result.mood.label)
```

---

## 6. Known limitations

- This is **not** a generative-AI video model, and it does not fetch
  anything from the internet — it assembles your own supplied photos/
  videos and locally generated backgrounds, never invented or downloaded
  footage.
- Your media folder applies to every song equally, all the time — there's
  no per-song override. A long track needing more shots than you have
  distinct files for simply reuses the same files round-robin (with
  videos getting a different random 2-4 second excerpt each time), so a
  small folder still works, just with less raw variety than a large one.
- Speech-to-text is used only to help detect a devotional theme for
  pacing; it has no effect on which visuals appear. Content-theme
  detection recognizes a curated devotional/spiritual vocabulary — other
  specific themes (weddings, travel vlogs, etc.) don't get any special
  pacing treatment, just the acoustic mood classifier's own cut speed.
- Rendering time scales with video length/resolution/fps and whether
  video clips (vs. stills) are used, since video decoding/re-encoding is
  more expensive than still-image Ken Burns animation. As a rough guide,
  a ~6-minute track at 1080p typically renders in the range of tens of
  minutes rather than hours; very long tracks (10+ minutes) or 4K output
  will take proportionally longer. Lowering the resolution/fps in
  Settings is the most effective way to speed up a render.
