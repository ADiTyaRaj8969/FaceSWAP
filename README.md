---
title: DeepFace Studio
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

---
&copy; Aditya Raj 2026
