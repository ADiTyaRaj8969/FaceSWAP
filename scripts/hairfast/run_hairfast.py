"""
CLI runner: hairstyle transfer with HairFastGAN on the local GPU.
Usage: python run_hairfast.py <face> <shape> <color> <out_png>
Must be run from the HairFastGAN repo dir (relative model paths).
"""
import sys
import faulthandler
import torch
import torchvision.transforms.functional as TF

# If anything blocks for >180s, dump every thread's stack so we can see the hang.
faulthandler.dump_traceback_later(180, repeat=True)

from hair_swap import HairFast, get_parser


def main():
    if len(sys.argv) < 5:
        print("usage: run_hairfast.py face shape color out", file=sys.stderr)
        return 2
    face, shape, color, out = sys.argv[1:5]

    args = get_parser().parse_args([])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = 1   # ease VRAM (default 3) so it co-exists with the Flask models
    print(f"[run] device={args.device}; building HairFast (loads models)...", flush=True)
    hair_fast = HairFast(args)
    print("[run] models loaded; running swap (align+embed+blend)...", flush=True)

    # align=True crops arbitrary photos to FFHQ-aligned faces internally.
    result = hair_fast.swap(face, shape, color, align=True)
    print("[run] swap finished; saving...", flush=True)
    if isinstance(result, (list, tuple)):
        result = result[0]
    img = TF.to_pil_image(result.detach().float().cpu().clamp(0, 1))
    img.save(out)
    print(f"[run_hairfast] saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
