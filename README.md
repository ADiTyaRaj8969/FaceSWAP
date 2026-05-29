---
title: DeepFace Studio
emoji: 🎭
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DeepFace Studio

Live demo: https://aditya-raj19-faceswap.hf.space

AI-powered face swap web application with seamless hair-to-neck blending, skin tone matching, and Firebase authentication.

## Features
- Live camera capture or image upload for source face
- Upload target face image
- Deep face swap using InsightFace inswapper model
- Hair-to-neck seamless blending (Laplacian + Poisson)
- Skin tone matching (CIE LAB colour transfer)
- Quality metrics after every swap
- Firebase Google Sign-in with 10-day session
- GPU + CPU support (auto-detected)

## Local Setup

### Requirements
- Python 3.10+
- Node.js 20+

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download the face swap model (~528 MB)
python scripts/download_models.py

# Build the React frontend
cd frontend
npm install
npm run build
cd ..

# Start the app
python web_app.py
```

Open http://localhost:5000

### Firebase Setup
1. Go to https://console.firebase.google.com
2. Create a new project
3. Add a Web App
4. Enable Google Sign-In under Authentication
5. Copy the config into `frontend/src/firebase.js`
6. Rebuild the frontend: `cd frontend && npm run build`

## Deployment (HuggingFace Spaces)
The app deploys automatically via GitHub Actions on every push to `main`.
The inswapper model is downloaded automatically at container startup.

## Tech Stack
- **Frontend**: React 18 + Vite + TailwindCSS + Framer Motion
- **Backend**: Python + Flask
- **AI Models**: InsightFace (inswapper_128), MediaPipe, OpenCV
- **Auth**: Firebase Google Sign-In

## Note
This tool is for educational purposes only. Use responsibly and only with images you own or have explicit consent to use.

## Pipeline

The swap runs a deliberately short, high-quality path — no heavy post-blending
that softens the face or smears the hairline:

1. **InsightFace `inswapper_128`** swaps the face (`paste_back` already blends the
   boundary; source skin tone is carried by the model).
2. **GFPGAN face restoration** recreates the facial detail lost in the model's
   128×128 internal resize — this is what removes the blur. Runs before the
   preview is encoded, so the on-screen result is sharp, not just the download.
3. **RealESRGAN ×4** upscales the result to ~4K for the downloadable PNG
   (Lanczos fallback if the SR weights aren't present).

All three models auto-detect and run on GPU (CUDA) when available.

## Full Feature List

### 🎭 Core Face Swap
- Deep face swap — InsightFace `inswapper_128` ONNX model, Haar-cascade `seamlessClone` fallback ([core/swapper.py](core/swapper.py))
- Multi-face targets — swaps every detected face
- Glasses/spectacle preservation — restores the target's eyewear region after the swap
- GFPGAN face restoration — fixes the 128×128 swap blur ([core/super_res.py](core/super_res.py))

### 🔍 Detection & Analysis
- Face detection — InsightFace `buffalo_l` with Haar fallback ([core/detector.py](core/detector.py))
- 468-point landmarks — MediaPipe Face Mesh ([core/landmarks.py](core/landmarks.py))
- Hair/skin/neck segmentation — BiSeNet or heuristic ([core/segmentor.py](core/segmentor.py))
- Skin-tone analysis — Fitzpatrick category + undertone from 5 face regions ([core/skin_tone.py](core/skin_tone.py))

### 📊 Quality Metrics (shown after every swap)
- Alignment, Blend quality, ΔE colour difference, Naturalness ([core/quality_checker.py](core/quality_checker.py))

### 🧰 Additional Modules (available for the full/batch pipelines)
- Seamless hair→face→neck blending ([core/neck_integrator.py](core/neck_integrator.py))
- Laplacian pyramid + Poisson seamless cloning ([core/blender.py](core/blender.py))
- Colour harmonization + boundary-lighting correction ([core/color_corrector.py](core/color_corrector.py))
- Face alignment — Procrustes transform ([core/aligner.py](core/aligner.py))
- Full pipeline with progress callbacks ([pipeline/full_pipeline.py](pipeline/full_pipeline.py)) and batch — one source → many targets ([pipeline/batch_pipeline.py](pipeline/batch_pipeline.py))

### 🖥️ Frontend (React 18 + Vite + Tailwind + Framer Motion)
- Landing + App pages with animated UI (Aurora, BlurText, SplitText, TiltedCard, Spotlight, CountUp, Magnet)
- Live camera capture or image upload for the source face
- Firebase Google Sign-In with 10-day session ([frontend/src/hooks/useAuth.js](frontend/src/hooks/useAuth.js))

### ⚙️ Backend & Infra
- Flask REST API — `/api/detect`, `/api/swap`, `/api/download` ([web_app.py](web_app.py))
- 4K PNG download of results
- Docker deployment (port 7860) + `startup.sh` auto-downloads models
- CI/CD — GitHub Actions: `ci.yml` (backend pytest + frontend build), `deploy.yml` (auto-push to HuggingFace Spaces on `main`)
- GPU + CPU auto-detection

Live demo: https://aditya-raj19-faceswap.hf.space

---
&copy; Aditya Raj 2026