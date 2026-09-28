# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Audio2Video.

Build with (from the repo root, inside the project's venv):

    pyinstaller packaging/audio2video.spec --noconfirm

Produces a "onedir" bundle at dist/audio2video/ containing Audio2Video.exe
plus every dependency it needs (a onedir build, rather than a single-file
--onefile exe, is used deliberately: this app pulls in several very heavy,
native-extension-laden libraries — PySide6/Qt, librosa/numba/llvmlite,
moviepy/imageio-ffmpeg, faster-whisper/ctranslate2/onnxruntime — and
unpacking all of that from a single-file exe into a temp directory on
every single launch would make startup noticeably slow and fragile. A
onedir folder is what the Inno Setup script in this same directory expects
to find at ../dist/audio2video relative to itself).

The PyInstaller hooks bundled with `moviepy`, `librosa` and `PySide6`
generally handle their own data files, but a few packages need an explicit
nudge (via collect_all/collect_data_files) because they either ship
non-Python data (onnxruntime capi, ctranslate2 shared libs, imageio-ffmpeg's
bundled ffmpeg binary) or use dynamic/lazy imports PyInstaller's static
analysis can't see (faster_whisper, tokenizers, PySide6 plugins).
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# --- Package metadata (dist-info) --------------------------------------
# Several dependencies (imageio, huggingface_hub, tokenizers, numba,
# librosa's own transitive deps like pooch/lazy_loader/decorator, etc.)
# call importlib.metadata.version(...) at import time to check their own
# or a related package's version. `collect_all()` below gathers each
# package's *data files and code*, but NOT its installed-distribution
# metadata (the .dist-info directory) -- that requires copy_metadata()
# specifically. Without it, the frozen exe crashes with
# `importlib.metadata.PackageNotFoundError` the first time any bundled
# library does a version lookup.
#
# Rather than hand-picking which of ~60 installed packages need this (new
# transitive dependencies could reintroduce the same crash later), copy
# metadata for every distribution actually installed in this build
# environment. This is cheap (a few KB of text per package) and
# eliminates the whole class of "No package metadata was found for X"
# errors, not just the one already seen for `imageio`.
try:
    from importlib.metadata import distributions as _distributions
except ImportError:  # pragma: no cover - Python <3.8 fallback, unused here
    _distributions = None

if _distributions is not None:
    for _dist in _distributions():
        _name = _dist.metadata.get("Name") if _dist.metadata else None
        if not _name:
            continue
        try:
            datas += copy_metadata(_name)
        except Exception:
            # A handful of internal/namespace packages have no proper
            # dist-info copy_metadata can read; skip those rather than
            # aborting the whole build.
            pass

# --- Packages that need their non-Python payload (native libs / data
# files / plugins) collected explicitly, since PyInstaller's default
# static analysis only follows plain Python imports. ------------------
for pkg in (
    "librosa",
    "moviepy",
    "imageio_ffmpeg",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "tokenizers",
    "huggingface_hub",
    "PySide6",
    "soundfile",
    "PIL",
):
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    except Exception:
        # Not fatal at spec-authoring time -- if a package isn't installed
        # in the environment running this spec, skip it rather than
        # aborting the whole build; the actual CI build environment
        # installs everything from requirements.txt first.
        pass

# librosa's numba/llvmlite JIT backend and audioread's format backends are
# both resolved dynamically at runtime and are easy for static analysis to
# miss entirely.
hiddenimports += collect_submodules("numba")
hiddenimports += collect_submodules("audioread")
hiddenimports += [
    "soundfile",
    "scipy.signal",
    "scipy.special",
    "numpy",
]

a = Analysis(
    ["../run.py"],
    pathex=["..", "."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "numpy.tests"],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Audio2Video",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="app_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="audio2video",
)
