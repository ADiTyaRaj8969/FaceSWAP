import os
import glob
import concurrent.futures
from pathlib import Path

import cv2
import numpy as np

from utils.image_io import load_image, save_image, resize_keep_aspect
from pipeline.full_pipeline import run_full_pipeline


def run_batch_pipeline(
    source_path: str,
    target_dir: str,
    output_dir: str,
    max_workers: int = 2,
    max_image_size: int = 1024,
    pipeline_kwargs: dict | None = None,
) -> list:
    """
    Process a single source face against all target images in target_dir.
    Saves results to output_dir. Returns list of (target_path, output_path, quality).
    """
    os.makedirs(output_dir, exist_ok=True)
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    target_paths = (
        glob.glob(os.path.join(target_dir, "*.jpg")) +
        glob.glob(os.path.join(target_dir, "*.jpeg")) +
        glob.glob(os.path.join(target_dir, "*.png")) +
        glob.glob(os.path.join(target_dir, "*.webp"))
    )

    source = resize_keep_aspect(load_image(source_path), max_image_size)
    results = []

    def _process_one(tgt_path):
        try:
            target = resize_keep_aspect(load_image(tgt_path), max_image_size)
            out = run_full_pipeline(source, target, **pipeline_kwargs)
            if "error" in out:
                return (tgt_path, None, out["error"])

            stem = Path(tgt_path).stem
            out_path = os.path.join(output_dir, f"{stem}_swapped.png")
            save_image(out["result"], out_path)
            return (tgt_path, out_path, out["quality"])
        except Exception as e:
            return (tgt_path, None, str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one, p): p for p in target_paths}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    return results
