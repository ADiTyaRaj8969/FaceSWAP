# GPU Enhancement Runbook — DeepFace Studio

Executed once `torch+cu121` **and** `onnxruntime-gpu` are installed. Goal: best
realistic output, working **both locally (RTX 4060)** and on **HF Pro GPU**.

---

## Phase 1 — Verify the GPU stack
1. `torch.cuda.is_available()` → True, `torch.cuda.get_device_name(0)` → RTX 4060.
2. `onnxruntime.get_available_providers()` → contains `CUDAExecutionProvider`.
3. Load InsightFace and confirm it initialises on `CUDAExecutionProvider`
   (check `_ORT_PROVIDERS` in `core/detector.py` is `[CUDA, CPU]`).
4. Confirm `core/super_res.py::_device()` and `core/head_swap.py::_device()`
   both return `cuda`.
5. **Fail-safe**: if onnxruntime-gpu can't find cuDNN, install matching cuDNN 9
   or pin `onnxruntime-gpu==1.19.2` (CUDA 12 + cuDNN 9).

## Phase 2 — Render + benchmark a real swap on GPU
1. Run the full pipeline (head → face → GFPGAN → skin-match → hair → blend →
   RealESRGAN ×4) on a real source + a location target.
2. Save the output, **view it**, and time it (expect a few seconds vs ~60s CPU).
3. Confirm only the main subject is swapped (no background/poster faces).

## Phase 3 — Quality tuning (now that there's GPU headroom)
1. **Working resolution**: raise `resize_keep_aspect(..., 1024)` in `web_app.py`
   to **1280–1536** (GPU can handle it → sharper swap, better landmarks).
2. **RealESRGAN ×4**: real GPU upscale for the download (currently Lanczos on
   CPU). Tune `realesrgan_weight` (~0.5) so skin isn't plastic.
3. **GFPGAN**: confirm it runs on GPU; keep `upscale=1` (RealESRGAN does scaling).
4. **Skin tone**: keep `match_skin_to_source` at 0.75; re-evaluate visually.
5. **Blend**: keep Laplacian 4-level over the face mask; check no hard seam.
6. Re-render and visually compare before/after each change (no guessing).

## Phase 4 — Real hair swap (HairFastGAN, GPU)
1. `python scripts/setup_hairfast.py` → clones `external/HairFastGAN` + weights,
   patches ops, sets up the MSVC/CUDA launcher.
2. `pip install git+https://github.com/openai/CLIP.git face_alignment lpips kornia`.
3. Set `HAIRFAST_LOCAL=1`; `core/hair_transfer.py` auto-uses the local GPU path.
4. Wire it into the pipeline: after the face swap, transfer the **source's
   hairstyle** onto the result (real hair, not the BiSeNet warp).
5. Render + view; first call is slow (op compile), then ~15–20s on the 4060.
6. **VRAM guard**: 8GB is tight — run HairFastGAN, free InsightFace/GFPGAN
   between stages if needed; fall back gracefully if OOM.

## Phase 5 — Automated test harness (the "100 tests per location")
Feasible on GPU (~3-5s/swap). 100 × 19 locations × 2 genders ≈ 3,800 swaps ≈
3-5 hrs on the RTX 4060 — run as a batch job.

**5a. Source face dataset**
- Need a folder of diverse SOURCE faces: light / medium / deep skin tones,
  round / oval / long face shapes, short / long / curly hair, with / without
  glasses and beard, balanced male + female.
- Source options: a face dataset (e.g. FFHQ samples / a provided folder).
  Drop them in `tests/faces/male/` and `tests/faces/female/`. The harness
  scales to however many are provided (10, 50, 100…).

**5b. Build `scripts/batch_test.py`**
- For each gender → each location → each source face: run the full GPU pipeline.
- Save outputs to `tests/output/<gender>/<location>/<source>.jpg`.
- Build a **contact sheet (grid)** per location for quick visual QA.
- Compute metrics per swap via `core/quality_checker.py`
  (alignment, blend, ΔE, naturalness) → write `tests/report.csv`.

**5c. Analyse + tune**
- Aggregate metrics: flag locations / source-types with low scores
  (e.g. deep skin tone + bright location, or long-hair sources).
- **View** the worst cases; tune the specific stage (skin-match strength,
  blend, head-swap region, hair) and re-run that subset.
- Iterate until the score distribution + visual quality are consistently good
  across all skin tones, face shapes and hairstyles.

**5d. Reality on "fine-tune the model itself"**
- The `inswapper_128` weights can't be fine-tuned here (no training pipeline /
  dataset / time). What we *can* tune is the **pipeline around it** (resolution,
  head/face/hair order, skin-match, blend, RealESRGAN, HairFastGAN). The harness
  drives that tuning with real numbers + visuals. For a genuinely different
  identity model, the upgrade path is a stronger swapper (e.g. SimSwap-HQ /
  inswapper-512 / a GPU diffusion swapper) — also GPU-only.

## Phase 6 — GPU deploy for HF Pro
1. `requirements-gpu.txt`: `torch==2.3.0+cu121`, `torchvision==0.18.0+cu121`
   (cu121 index), `onnxruntime-gpu==1.19.2`, + the rest of the deps.
2. `Dockerfile.gpu`: base `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
   (CUDA 12 + cuDNN 9), install Python 3.10, torch cu121, onnxruntime-gpu, deps,
   build the React frontend (stage 1), copy app, `CMD ./startup.sh`.
3. README: note "switch HF Space to GPU hardware + use Dockerfile.gpu".
4. Keep the CPU `Dockerfile` as fallback so nothing breaks before the upgrade.

## Phase 7 — Ship
1. `npm run build` (frontend), syntax-check backend, run `pytest`.
2. Commit + push; confirm CI green and HF deploy green.
3. On HF: switch Space to **GPU hardware**, point to `Dockerfile.gpu`, set the
   existing secrets; verify a live GPU swap.

---

### Guardrails
- **Verify every change by rendering + viewing the image** — no blind tuning.
- Keep CPU fallback intact (so the app still runs without a GPU).
- Don't break the green HF deploy until the GPU Space is confirmed working.
