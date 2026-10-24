"""Build the Recast native library (native/recast_c + vendored Recast) for bsp_waypointer.recast_native.

    python tools/build_recast.py            # MSVC on Windows (found with vswhere), c++ elsewhere

Output: src/bsp_waypointer/_native/recast_c.dll (Windows) or librecast_c.so, which is gitignored.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECAST = ROOT / "native" / "recastnavigation" / "Recast"
WRAPPER = ROOT / "native" / "recast_c" / "recast_c.cpp"
OUT_DIR = ROOT / "src" / "bsp_waypointer" / "_native"


def sources():
    return [WRAPPER] + sorted((RECAST / "Source").glob("*.cpp"))


def vcvars64() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        sys.exit("vswhere.exe not found: install Visual Studio with the C++ workload")
    inst = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True).stdout.strip()
    bat = Path(inst) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not bat.exists():
        sys.exit("vcvars64.bat not found under %r" % inst)
    return bat


def build_windows() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dll = OUT_DIR / "recast_c.dll"
    with tempfile.TemporaryDirectory() as obj:
        # cmd's quoting is its own, so the command goes through a script rather than an argument
        srcs = " ".join('"%s"' % s for s in sources())
        script = Path(obj) / "build.cmd"
        script.write_text('@call "%s" >nul\r\ncl /nologo /O2 /EHsc /MT /LD /DNDEBUG /I"%s" /Fo"%s\\\\" %s '
                          '/Fe"%s" /link /NOLOGO\r\n' % (vcvars64(), RECAST / "Include", obj, srcs, dll))
        r = subprocess.run(["cmd", "/d", "/c", str(script)], capture_output=True, text=True, cwd=obj)
        if r.returncode != 0 or not dll.exists():
            sys.exit("build failed:\n" + r.stdout[-4000:] + r.stderr[-2000:])
    for junk in (dll.with_suffix(".lib"), dll.with_suffix(".exp")):
        if junk.exists():
            junk.unlink()
    return dll


def build_posix() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    so = OUT_DIR / "librecast_c.so"
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not cxx:
        sys.exit("no C++ compiler (set CXX)")
    subprocess.run([cxx, "-O2", "-shared", "-fPIC", "-DNDEBUG", "-I", str(RECAST / "Include"),
                    *map(str, sources()), "-o", str(so)], check=True)
    return so


def main() -> int:
    out = build_windows() if os.name == "nt" else build_posix()
    print("built %s (%d bytes)" % (out, out.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
