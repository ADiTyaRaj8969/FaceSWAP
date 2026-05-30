"""
Set up the LOCAL GPU HairFastGAN hair-transfer backend (Windows + NVIDIA).

This vendors the HairFastGAN repo + weights under external/ (gitignored), patches
its CUDA ops for torch 2.x, links the weights into place, and installs the runner
scripts. After this, core/hair_transfer.py uses the local GPU automatically.

Requirements: git + git-lfs, an NVIDIA GPU with CUDA, Visual Studio Build Tools
(C++), and the project's Python env. Run from the project root:

    python scripts/setup_hairfast.py

Then install the extra model deps (one-time):

    pip install git+https://github.com/openai/CLIP.git face_alignment lpips kornia

On non-GPU / non-Windows hosts (e.g. HuggingFace CPU Spaces) this is unnecessary —
hair_transfer.py falls back to the hosted HairFastGAN Space automatically.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
REPO = EXT / "HairFastGAN"
WEIGHTS = EXT / "HF_weights"
RUNNERS = ROOT / "scripts" / "hairfast"

CODE_URL = "https://github.com/AIRI-Institute/HairFastGAN"
WEIGHTS_URL = "https://huggingface.co/AIRI-Institute/HairFastGAN"


def run(cmd, **kw):
    print(f"  $ {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=True, **kw)


def clone_code():
    if (REPO / "hair_swap.py").exists():
        print("[1/5] code repo already present — skip")
        return
    print("[1/5] cloning HairFastGAN code...")
    EXT.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", CODE_URL, str(REPO)])


def clone_weights():
    if (WEIGHTS / "pretrained_models").exists():
        print("[2/5] weights repo already present — running git lfs pull to finish")
        run(["git", "lfs", "pull"], cwd=str(WEIGHTS))
        return
    print("[2/5] cloning HairFastGAN weights (~7GB, git-lfs)...")
    run(["git", "clone", WEIGHTS_URL, str(WEIGHTS)])
    run(["git", "lfs", "pull"], cwd=str(WEIGHTS))


def patch_ops():
    """torch >=2 removed AT_CHECK and Tensor.data<T>(); rewrite the op sources."""
    print("[3/5] patching CUDA op sources for torch 2.x...")
    n = 0
    for p in REPO.rglob("*"):
        if p.suffix in (".cpp", ".cu") and p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            new = txt.replace("AT_CHECK", "TORCH_CHECK")
            new = re.sub(r"\.data<", ".data_ptr<", new)
            if new != txt:
                p.write_text(new, encoding="utf-8")
                n += 1
    print(f"      patched {n} files")


def link_weights():
    """Make pretrained_models/ and input/ available inside the repo dir."""
    print("[4/5] linking weights into the repo...")
    for name in ("pretrained_models", "input"):
        link = REPO / name
        target = WEIGHTS / name
        if link.exists() or link.is_symlink():
            continue
        if not target.exists():
            print(f"      WARNING: {target} missing")
            continue
        try:
            # Windows junction (no admin needed); symlink elsewhere.
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               check=True, capture_output=True)
            else:
                os.symlink(target, link)
            print(f"      linked {name}")
        except Exception as e:
            print(f"      link failed ({e}); copying instead...")
            shutil.copytree(target, link)


def install_runners():
    print("[5/5] installing runner scripts...")
    for f in ("run_hairfast.py", "run_hairfast.bat"):
        shutil.copy(RUNNERS / f, REPO / f)
    print("      done")


def main():
    print("HairFastGAN local-GPU setup")
    print("=" * 40)
    try:
        clone_code()
        clone_weights()
        patch_ops()
        link_weights()
        install_runners()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {e}\nSee the script header for requirements.")
        return 1
    print("\nSetup complete. Install the extra deps if you haven't:")
    print("  pip install git+https://github.com/openai/CLIP.git face_alignment lpips kornia")
    print("\nThe first hair swap is slow (one-time op compile + CLIP/vgg download);")
    print("subsequent swaps are ~18s. hair_transfer.py will use the local GPU now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
