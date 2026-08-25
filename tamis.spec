# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Tamis.

Build a standalone, single-file executable with:
    pyinstaller tamis.spec

Must be run separately on each target OS (Linux, Windows, macOS) --
PyInstaller does not cross-compile. See .github/workflows/release.yml for
the automated multi-platform build.
"""

from PyInstaller.utils.hooks import collect_all

# Everything MainWindow's Help menu can open. A doc reachable from the menu
# but missing here exists in a source checkout and not in the frozen build,
# so the omission only shows up in a release.
datas = [
    ("docs/face_recognition.html", "docs"),
    ("docs/sharpness.html", "docs"),
    ("docs/architecture.html", "docs"),
]
binaries = []
hiddenimports = []

# pillow-heif bundles a native libheif; collect_all pulls in its shared
# libraries and data files that PyInstaller's static analysis can't see.
for pkg in ("pillow_heif",):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Face recognition (torch/torchvision/facenet-pytorch) is only present in
# the build environment for the -cpu/-gpu release variants (see the
# `variant` axis in .github/workflows/release.yml's matrix) -- the lean
# variant never installs requirements-recognition.txt, so this guards the
# collection the same way the app itself guards the import
# (RECOGNITION_AVAILABLE in tamis/main_window.py). Each of these bundles
# its own native shared libraries (CUDA runtime libs for the GPU build in
# particular) that static analysis can't see, same reasoning as pillow_heif
# above.
try:
    import torch  # noqa: F401

    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False

if RECOGNITION_AVAILABLE:
    for pkg in ("torch", "torchvision", "facenet_pytorch"):
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports

# Aesthetic quality scoring (optional, see requirements-quality.txt). Guarded
# the same way, and separately from the block above: it is a different extra,
# so a build environment can legitimately have one and not the other.
# open_clip ships model configs as package data that static analysis cannot
# see, and timm is reached only through open_clip's registry -- without
# collect_all, the frozen build imports open_clip and then fails to find the
# model it was asked for.
try:
    import open_clip  # noqa: F401

    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False

if QUALITY_AVAILABLE:
    for pkg in ("open_clip", "timm"):
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Tamis only uses QtCore/QtGui/QtWidgets; excluding the rest keeps the
    # build smaller and skips their PyInstaller hooks (e.g. QtNetwork's hook
    # probes OpenSSL support at build time, which is unnecessary here and can
    # be flaky on systems with multiple conflicting OpenSSL/libbrotli builds).
    excludes=[
        "PySide6.QtNetwork",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtPdf",
        "PySide6.QtMultimedia",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Tamis",
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
)
