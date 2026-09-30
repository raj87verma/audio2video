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
       │                    detected, and extracts a *specific* named deity
       │                    if present (e.g. "Khatu Shyam") — see
       │                    "Content-aware visuals" below
       ▼
4. Shot planning                          → beat-aligned list of shots
       │
       ▼
5. Media fetch  (your local-only media folder, if configured, for every
       │         shot in every song; else your own deity-specific photos/
       │         videos, if any, for every shot in a matching song; else
       │         Wikimedia Commons for named deities; else Pexels /
       │         Pixabay, optional)  → photo/video per shot
       │        ↳ falls back to generated visuals if none of the above match
       ▼
6. Video assembly         (MoviePy / FFmpeg)
       ↳ Ken Burns zoom/pan on stills, crossfades, mood color grading
       ↳ synthetic (+ optional Freesound) SFX layered under your audio
       ▼
   finished .mp4 (+ a credits.txt if Wikimedia media was used)
```

### Local-only mode: use ONLY your own media, for every song (highest priority)

If you'd rather Audio2Video never call out to Wikimedia/Pexels/Pixabay at
all — for any song, not just devotional ones — point it at a folder of
your own royalty-free photos/videos and it will use **only** that folder,
permanently, until you turn it off again.

**How to set it up:** open the app's **Settings** tab → find "Local-Only
Media Folder" → click **Browse...** and pick any folder on your computer
(it doesn't need to exist yet, or be empty — pick wherever makes sense for
you). Audio2Video automatically creates two subfolders inside it:

```
<the folder you picked>/
  images/    <- put your photos here
  videos/    <- put your video clips here
```

Drop any number of files into `images/` and `videos/` (supported formats:
`.jpg`/`.jpeg`/`.png`/`.bmp`/`.webp` for images, `.mp4`/`.mov`/`.webm`/
`.mkv`/`.avi`/`.m4v` for videos). The Settings tab shows a live count of
how many of each it finds.

**Once a folder is set here, this is a permanent, global switch:**
- Every song you process from then on uses ONLY the files in this folder
  — Wikimedia Commons, Pexels, and Pixabay are never contacted again, for
  any song, regardless of its filename/lyrics/mood.
- If the folder happens to be empty (or you haven't added files yet),
  shots simply use the generated animated backgrounds instead — it does
  **not** fall back to any online source, ever, while this is set.
- Click **Clear** in Settings to turn this off and go back to the normal
  Wikimedia/Pexels/Pixabay behavior described below.

What happens with your files (same treatment as the deity-specific folder
below):
- **Videos** are used in short, randomly varied 2-4 second excerpts per
  shot — never the whole clip at once. If a shot needs more than 4
  seconds (shot length varies with the song's tempo/mood), that short
  excerpt simply loops to fill the remaining time rather than jumping to
  a different part of the video mid-shot.
- **Photos** get the same Ken Burns pan/zoom and color grading as any
  other still image, plus a subtle animated glow + twinkling sparkle
  effect on top, so they don't look flat/static next to the rest of the
  video.
- If you have both images and videos in the folder, shots mix between
  them (not "all video until it runs out, then all photos") so a song
  actually uses the variety you've supplied.

**How this relates to the deity-specific folder below:** this local-only
folder is a single, simple, always-on switch that applies to every song.
The "Use your own photos/videos" feature described next is lighter-weight
— it only activates for a song about a specific *recognized deity*, and
still falls back online if that deity's folder happens to be empty. If
you configure **both**, this local-only folder takes priority for
everything, and the deity-specific folder is never consulted at all.

Only use media you actually have the rights to use for this (your own
photos/videos, or ones whose license explicitly permits it) — this
feature doesn't check licensing for you, since it's your own supplied
files.

### Use your own photos/videos (per recognized deity)

Even with deity-specific Wikimedia footage (below), a song's shots still
cycle through a *mix* of the deity's own visuals and generic devotional
terms ("hindu temple", "diya lamp", ...) for variety — so a few shots out
of a longer video can still end up looking generic rather than specific to
your song's subject. If you have your own royalty-free photos/videos —
of Khatu Shyam, or any other deity in the recognized list — you can supply
them directly, and Audio2Video will use **only your own files, for every
single shot**, skipping Wikimedia/Pexels/Pixabay entirely for that song.

**Where to put them:** open the app's **Settings** tab → find "Your Own
Deity Photos/Videos" → the folder path is shown there (click **Open
Folder** to jump straight to it in your file manager). By default this is:

- Windows: `%USERPROFILE%\.audio2video\user_media\<deity>\`
- macOS/Linux: `~/.audio2video/user_media/<deity>/`

Inside `user_media/` there's already one ready-made subfolder per
recognized deity (`khatu_shyam/`, `krishna/`, `hanuman/`, `shiva/`, ...,
matching the same list in "Deity-specific footage" below), each containing
a short `README.txt`. Just drop your files into the matching subfolder —
e.g. for Khatu Shyam, into `user_media/khatu_shyam/`. Supported formats:
images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) and videos (`.mp4`,
`.mov`, `.webm`, `.mkv`, `.avi`, `.m4v`).

What happens with your files:
- **Videos** are used in short excerpts (matching each shot's length, a
  few seconds at a time). Since a song usually has many more shots than
  you're likely to supply distinct video files for, the *same* video gets
  reused across multiple shots — each reuse picks a different, randomly
  chosen part of the clip, so a single video still gives visual variety
  across a whole song instead of looping the same few seconds every time.
- **Photos** get the same Ken Burns pan/zoom and color grading as any
  other still image, plus a subtle animated glow + twinkling sparkle
  effect layered on top, so a static personal photo doesn't look
  completely motionless next to the rest of the video.
- If the matching folder is empty, nothing changes — the song falls back
  to Wikimedia Commons / Pexels / Pixabay / generated visuals exactly as
  described below, with no extra configuration needed.

Only use media you actually have the rights to use for this (your own
photos/videos, or ones whose license explicitly permits it) — this
feature doesn't check licensing for you, since it's your own supplied
files.

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

#### Deity-specific footage via Wikimedia Commons

Generic devotional keywords ("temple", "diya lamp", "prayer") already come
from Pexels/Pixabay, but those libraries have **essentially no coverage of
named deities** — searching "Khatu Shyam" (or most other specific deity/
temple names) on either returns nothing. So if a devotional song names a
*specific* deity — recognized from the filename or transcribed lyrics
(currently covers Khatu Shyam, Krishna, Radha Krishna, Ram, Sita, Hanuman,
Shiva, Durga, Kali, Amba Mata, Lakshmi, Saraswati, Ganesh, Vishnu, Sai
Baba, Balaji, Vaishno Devi, in both English and Devanagari spellings) —
the app checks **Wikimedia Commons** first for that deity specifically.
Wikimedia is a free, crowd-sourced media library (the same organization
behind Wikipedia) populated by devotees photographing/filming their own
temples and ceremonies, so it has real, on-theme coverage that stock-photo
sites don't. No sign-up or API key is required for this. It only ever
activates for this narrow, named-deity case — every other keyword still
goes to Pexels/Pixabay as before. It can be turned off in **Settings** if
you'd rather skip it (see the licensing note below for why you might).

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
| **Wikimedia Commons** | deity/temple-specific photos & videos | Free, **no sign-up or key needed at all** | n/a — used automatically, toggle in Settings |

Steps for Pexels/Pixabay/Freesound (Wikimedia needs no key at all — see its
row above and the "Deity-specific footage" section below):
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
- **Use Wikimedia Commons for named deities/temples** — on by default,
  requires no key/sign-up. Turn off if you'd rather every shot come only
  from Pexels/Pixabay/generated visuals, or to avoid the attribution
  requirement described in the licensing section below.
- **Your Own Deity Photos/Videos folder** — a read-only path display plus
  an **Open Folder** button; see "Use your own photos/videos (per
  recognized deity)" above. There's no on/off toggle for this one — it's
  simply used whenever the matching deity subfolder has at least one
  supported file in it (and only when the local-only folder below isn't
  set).
- **Local-Only Media Folder** — an editable path field plus **Browse...**,
  **Clear**, and **Open Folder** buttons, and a live image/video count;
  see "Local-only mode" above. Empty (the default) means this is off and
  the app behaves as described everywhere else in this README. Setting
  a folder here is a **permanent, global** switch — it takes priority
  over every other visual source, for every song, until you click
  **Clear**.

---

## 4. What "SFX and VFX added automatically" actually means here

- **VFX**: Ken Burns zoom/pan on still images, mood-based color grading
  (5 built-in "grades" — cool/contrasty, warm/vibrant, moody/desaturated,
  soft pastel, neutral cinematic), crossfade transitions between shots, and
  — whenever no stock footage matches — a fully generated animated
  background (gradient + drifting particles + a live waveform visualizer
  reacting to your track's actual loudness). Any photos you supply
  yourself (via either the local-only folder or the per-deity folder)
  additionally get a subtle animated glow + twinkling sparkle overlay —
  see "Local-only mode" and "Use your own photos/videos" above.
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
- **Wikimedia Commons media is different from the other sources above: it
  almost always requires attribution.** Nearly everything on Commons is
  published under a Creative Commons license (typically CC BY 4.0 or
  CC BY-SA 4.0) that legally requires crediting the original author and
  linking to the license wherever the work is used. Unlike Pexels/Pixabay,
  this is **not optional** if you use Wikimedia media (which only happens
  for the named-deity case described above). To make this easy: whenever
  a render actually uses Wikimedia media, the app automatically writes a
  `<video-name>_credits.txt` file next to the finished video, listing
  every item's title, author, license (with a link), and source page.
  **Copy that text into the video's description** when you publish it
  (YouTube, Instagram, etc.) to stay compliant. If you'd rather not deal
  with this at all, turn off "Use Wikimedia Commons" in Settings — the app
  will just use Pexels/Pixabay/generated visuals instead, exactly as it
  did before this feature existed.
- The **procedurally generated visuals and synthetic SFX** produced by
  this app are code-generated locally and carry no third-party licensing
  restrictions.
- Aside from the automatic Wikimedia credits file above, this tool doesn't
  remove or add any attribution automatically — if a provider's terms
  require attribution for a specific use case, add it yourself.

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
│   │   ├── wikimedia_fetcher.py Wikimedia Commons client (named-deity footage)
│   │   ├── user_media.py        scans the user's own supplied deity photos/videos
│   │   ├── local_media.py       scans the permanent local-only-media folder
│   │   ├── credits.py           builds the Wikimedia attribution credits file
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
- Named-deity detection (Wikimedia Commons) only recognizes a curated list
  of deity names (see "Deity-specific footage" above); a devotional song
  naming a deity outside that list still falls back to generic devotional
  keywords via Pexels/Pixabay, same as before this feature existed.
  Wikimedia's search/coverage also varies by deity — very well-known
  deities (Krishna, Hanuman, Shiva, ...) tend to have far more real photos
  and video than more regional/local ones, so visual variety per deity is
  not guaranteed to be equally rich across the list.
- Your own supplied media (via either the per-deity folder or the
  local-only folder — see the two "Use your own photos/videos" /
  "Local-only mode" sections above) is an all-or-nothing switch: as soon
  as it applies, *every* shot uses only your own media, with no way to
  mix in Wikimedia/Pexels/Pixabay visuals alongside it for extra variety.
  For the local-only folder this is global (every song, permanently,
  until you click Clear); for the per-deity folder it's just per-song. If
  you want a mix, that currently means manually supplying enough of your
  own variety to cover the whole song.
