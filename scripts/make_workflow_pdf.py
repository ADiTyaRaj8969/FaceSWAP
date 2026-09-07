"""
Generate docs/WORKFLOW.pdf — how DeepFace Studio actually works end to end.

Kept in the repo so the document can be regenerated after the pipeline changes,
instead of drifting away from the code the way a hand-drawn diagram would.

Run:  python scripts/make_workflow_pdf.py
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

NAVY = (0.06, 0.09, 0.16)
TEAL = (0.05, 0.58, 0.53)
SLATE = (0.28, 0.33, 0.41)
LIGHT = (0.95, 0.96, 0.97)
BORDER = (0.80, 0.84, 0.88)
AMBER = (0.85, 0.47, 0.05)

W, H = A4
M = 18 * mm


def header(c, title, subtitle=None):
    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(M, H - M - 4, title)
    if subtitle:
        c.setFillColorRGB(*SLATE)
        c.setFont("Helvetica", 9.5)
        c.drawString(M, H - M - 19, subtitle)
    c.setStrokeColorRGB(*TEAL)
    c.setLineWidth(2)
    c.line(M, H - M - 26, W - M, H - M - 26)
    return H - M - 42


def footer(c, page, total):
    c.setFillColorRGB(*SLATE)
    c.setFont("Helvetica", 7.5)
    c.drawString(M, 12 * mm, "DeepFace Studio — project workflow")
    c.drawRightString(W - M, 12 * mm, f"{page} / {total}")


def box_height(body=None):
    """Height that fits a title plus these body lines, with even padding."""
    return 24 + len(body or []) * 10 + 2


def box(c, x, y, w, h, title, body=None, fill=LIGHT, accent=TEAL, tsize=9.5):
    """
    Rounded box with an accent bar; returns the y of its bottom edge.
    Pass h=None to size it from the body, so adding a line can't overflow it.
    """
    if h is None:
        h = box_height(body)
    c.setFillColorRGB(*fill)
    c.setStrokeColorRGB(*BORDER)
    c.setLineWidth(0.8)
    c.roundRect(x, y - h, w, h, 3, stroke=1, fill=1)
    c.setFillColorRGB(*accent)
    c.roundRect(x, y - h, 3, h, 1.5, stroke=0, fill=1)

    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", tsize)
    c.drawString(x + 9, y - 12.5, title)
    if body:
        c.setFillColorRGB(*SLATE)
        c.setFont("Helvetica", 8.2)
        ty = y - 24
        for line in body:
            c.drawString(x + 9, ty, line)
            ty -= 10
    return y - h


def arrow(c, x, y1, y2, label=None):
    c.setStrokeColorRGB(*TEAL)
    c.setLineWidth(1.2)
    c.line(x, y1, x, y2)
    c.setFillColorRGB(*TEAL)
    c.setLineWidth(0)
    p = c.beginPath()
    p.moveTo(x, y2)
    p.lineTo(x - 3.2, y2 + 6)
    p.lineTo(x + 3.2, y2 + 6)
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    if label:
        c.setFillColorRGB(*SLATE)
        c.setFont("Helvetica-Oblique", 7.5)
        c.drawString(x + 8, (y1 + y2) / 2 - 2, label)


def bullets(c, x, y, items, width_note=None):
    c.setFont("Helvetica", 9)
    for it in items:
        c.setFillColorRGB(*TEAL)
        c.drawString(x, y, "•")
        c.setFillColorRGB(*SLATE)
        c.drawString(x + 10, y, it)
        y -= 12.5
    return y


# ── page 1 — overview ────────────────────────────────────────────────────────
def page_overview(c):
    y = header(c, "DeepFace Studio",
               "AI face swap that places a visitor into a real campus photograph")

    c.setFillColorRGB(*SLATE)
    c.setFont("Helvetica", 9.2)
    for line in [
        "A visitor enters their name, uploads or captures a photo, and picks a campus location.",
        "Their face is swapped into the curated photograph for that location, then returned as a",
        "preview plus a high-resolution download.",
    ]:
        c.drawString(M, y, line)
        y -= 12.5
    y -= 8

    y = box(c, M, y, W - 2 * M, None, "Tech stack", [
        "Frontend   React 18 + Vite + TailwindCSS + Framer Motion (SPA, served by Flask)",
        "Backend    Python + Flask, background worker threads for the swap pipeline",
        "Storage    Supabase (location photographs) with a local Location/ folder fallback",
        "Auth       Firebase Google Sign-In — gates the Control Panel to an email allowlist",
    ]) - 14

    y = box(c, M, y, W - 2 * M, None, "Models", [
        "buffalo_l (InsightFace)      face detection + 5-point landmarks",
        "inswapper_128 (InsightFace)  the identity swap itself, internally 128x128",
        "GFPGAN v1.4                  restores facial detail lost to that 128px bottleneck",
        "RealESRGAN x4plus            super-resolution, applied to the head region",
        "BiSeNet (facexlib)           hair / skin / neck segmentation",
        "HairFastGAN (hosted Space)   optional hairstyle transfer — needs HF_TOKEN",
    ]) - 14

    y = box(c, M, y, W - 2 * M, None, "Hosting", [
        "Hugging Face Space  aditya-rAj19/FaceSWAP  (Docker SDK, port 7860)",
        "Live                https://aditya-raj19-faceswap.hf.space",
        "Hardware            cpu-basic (free tier); CUDA is used automatically when present",
        "Deploy              every push to main, via GitHub Actions",
    ]) - 14

    box(c, M, y, W - 2 * M, None, "Why the pipeline is built the way it is", [
        "The swap model works at 128x128, so the head is the only part of the frame that",
        "loses real detail. Everything else is taken from the original photograph untouched,",
        "and super-resolution is spent on the head alone rather than the whole image.",
    ], fill=(0.98, 0.96, 0.90), accent=AMBER)


# ── page 2 — architecture ────────────────────────────────────────────────────
def page_architecture(c):
    y = header(c, "System architecture", "What talks to what, at request time")

    cw = W - 2 * M
    y = box(c, M, y, cw, None, "Browser — React SPA", [
        "/ landing    /app swap page (Head Swap toggle)    /admin Control Panel",
    ])
    arrow(c, W / 2, y, y - 22, "HTTPS")
    y -= 22

    y = box(c, M, y, cw, None, "Flask (web_app.py)", [
        "Serves the built SPA and the JSON API. Static assets get explicit MIME types",
        "because the slim Python image ships an incomplete MIME database.",
    ])
    arrow(c, W / 2, y, y - 22, "submit / poll")
    y -= 22

    half = (cw - 10) / 2
    y2 = box(c, M, y, half, None, "Job store (in memory)", [
        "One swap at a time — the",
        "models already saturate the",
        "2 vCPUs. Results expire after",
        "20 minutes and are capped.",
    ])
    box(c, M + half + 10, y, half, None, "Worker thread", [
        "_perform_swap() runs the",
        "pipeline with no request",
        "context and reports progress",
        "back to the job.",
    ])
    y = y2
    arrow(c, W / 2, y, y - 22)
    y -= 22

    y = box(c, M, y, cw, None, "core/ pipeline", [
        "detector · swapper · super_res · head_swap · segmentor · blender · hair_transfer",
    ])
    y -= 16

    y2 = box(c, M, y, half, None, "Supabase", [
        "Location photographs per",
        "gender. Falls back to the",
        "Location/ folder when unset.",
    ], accent=SLATE)
    box(c, M + half + 10, y, half, None, "Firebase", [
        "Google Sign-In for the",
        "Control Panel; ID tokens are",
        "verified server-side too.",
    ], accent=SLATE)

    c.setFillColorRGB(*SLATE)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(M, y2 - 16, "Nothing is written to disk during a swap — images are decoded, processed and returned in memory.")


# ── page 3 — request flow ────────────────────────────────────────────────────
def page_request(c):
    y = header(c, "Request workflow", "Why a swap is a job rather than a single request")

    box(c, M, y, W - 2 * M, None, "The constraint", [
        "Full-quality output takes minutes on the free CPU tier — head super-resolution alone",
        "is about six. No HTTP request survives that, so the swap cannot be synchronous.",
    ], fill=(0.98, 0.96, 0.90), accent=AMBER)
    y -= 62

    steps = [
        ("1  POST /api/swap", [
            "Validates and decodes the source, resolves the target from gender + location,",
            "creates a job and returns 202 with a job_id. Returns in about 1.5 seconds.",
        ]),
        ("2  Worker thread picks up the job", [
            "Runs the pipeline, updating progress and a message at each stage.",
        ]),
        ("3  GET /api/swap/status/<job_id>", [
            "Polled every 2 seconds. Returns state, progress and message while running.",
        ]),
        ("4  state = done", [
            "The same payload the endpoint always returned: result_image, download_image,",
            "quality, delta_e, src_tone, tgt_tone, warnings, name.",
        ]),
    ]
    for i, (title, body) in enumerate(steps):
        y = box(c, M, y, W - 2 * M, None, title, body)
        if i < len(steps) - 1:
            arrow(c, W / 2, y, y - 18)
            y -= 18
    y -= 20

    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(M, y, "In the UI")
    y -= 16
    bullets(c, M, y, [
        "The progress bar follows the server's real stage, not a timer animating a guess.",
        "The motivational captions keep rotating on their own — the wait is longer now.",
        "Failures surface the server's message instead of failing silently.",
    ])


# ── page 4 — the pipeline ────────────────────────────────────────────────────
def page_pipeline(c):
    y = header(c, "The swap pipeline",
               "Stage by stage, with measured CPU cost on a 1086x1448 photograph")

    c.setFillColorRGB(*SLATE)
    c.setFont("Helvetica", 8.6)
    c.drawString(M, y, "Working resolution is 1024 on CPU and 1536 on CUDA. Optional stages are marked.")
    y -= 16

    rows = [
        ("Prepare", "Resize; pad a close-up until detectable; de-roll a tilted selfie so eyes are level.", "~7s"),
        ("Detect", "buffalo_l on source and target; retries with CLAHE, then Haar.", "~1s"),
        ("Head swap  (opt-in)", "Transplants the whole head — shape and hair — instead of only the face.", "varies"),
        ("Face swap", "inswapper_128 puts the source identity on the main target face.", "~19s"),
        ("Restore", "GFPGAN rebuilds the detail the 128px swap threw away.", "~55s"),
        ("Blend seam", "Laplacian pyramid over the face boundary. Must run before hair.", "~14s"),
        ("Hair  (opt-in)", "HairFastGAN only. No crude fallback — it smeared a blob over the head.", "~30s"),
        ("Glasses  (opt-in)", "Off by default: it disfigured sources who wear glasses.", "~4s"),
        ("Skin tone", "Drives face, neck, arms and hands to the source complexion.", "~3s"),
        ("Score", "Alignment, blend, delta-E, naturalness — measured before grain is added.", "~2s"),
        ("Harmonise", "Adds the photo's grain over the parsed head so it isn't GAN-smooth.", "~1s"),
        ("Composite", "Pastes only what changed back onto the untouched full-resolution photo.", "~0.2s"),
        ("Enhance", "RealESRGAN on the head crop, Lanczos elsewhere, feathered between.", "~370s"),
    ]

    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", 8.4)
    c.drawString(M + 4, y, "STAGE")
    c.drawString(M + 108, y, "WHAT IT DOES")
    c.drawRightString(W - M - 4, y, "CPU")
    y -= 4
    c.setStrokeColorRGB(*BORDER)
    c.setLineWidth(0.6)
    c.line(M, y, W - M, y)
    y -= 13

    for i, (name, desc, cost) in enumerate(rows):
        if i % 2 == 0:
            c.setFillColorRGB(0.975, 0.98, 0.985)
            c.rect(M, y - 4, W - 2 * M, 15, stroke=0, fill=1)
        opt = "(opt-in)" in name
        c.setFillColorRGB(*(SLATE if opt else NAVY))
        c.setFont("Helvetica-Bold", 8.2)
        c.drawString(M + 4, y, name)
        c.setFillColorRGB(*SLATE)
        c.setFont("Helvetica", 8)
        c.drawString(M + 108, y, desc)
        c.setFillColorRGB(*(AMBER if cost == "~370s" else SLATE))
        c.setFont("Helvetica-Bold" if cost == "~370s" else "Helvetica", 8)
        c.drawRightString(W - M - 4, y, cost)
        y -= 15

    y -= 12
    box(c, M, y, W - 2 * M, None, "The two decisions that matter most", [
        "Composite, not upscale — the background and body are never touched by the swap, so",
        "they are taken from the original photograph. Background sharpness went from 34% of",
        "the original to 100%, and the delivered image is native resolution rather than a",
        "downscaled frame blown back up.",
    ], fill=(0.94, 0.98, 0.97))


# ── page 5 — deployment ──────────────────────────────────────────────────────
def page_deploy(c):
    y = header(c, "Deployment", "From a push on main to a running Space")

    seq = [
        ("git push origin main", ["The default branch is the deploy trigger."]),
        ("GitHub Actions", [
            "ci.yml       pytest on a lightweight dependency subset, plus a frontend build",
            "deploy.yml   force-pushes the repository to the Hugging Face Space",
        ]),
        ("Hugging Face Space", [
            "Rebuilds the Docker image on the new commit.",
        ]),
        ("Docker build", [
            "Stage 1  node:20-slim builds the React app into static/react",
            "Stage 2  python:3.10-slim installs CPU torch, then requirements-deploy.txt",
        ]),
        ("startup.sh", [
            "Downloads model weights in the BACKGROUND and starts Flask immediately —",
            "a container that doesn't answer on its port within ~60s is killed.",
        ]),
        ("Flask on :7860", [
            "Debug off; models pre-warm on a daemon thread so the port binds at once.",
        ]),
    ]
    for i, (title, body) in enumerate(seq):
        y = box(c, M, y, W - 2 * M, None, title, body)
        if i < len(seq) - 1:
            arrow(c, W / 2, y, y - 16)
            y -= 16
    y -= 22

    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(M, y, "Configuration held outside the repository")
    y -= 16
    y = bullets(c, M, y, [
        "HF_TOKEN — GitHub secret for deploying; also lifts the HairFastGAN GPU quota.",
        "SUPABASE_URL / SUPABASE_SERVICE_KEY / SUPABASE_BUCKET — location photographs.",
        "ALLOWED_ADMIN_EMAILS / PRIMARY_ADMIN_EMAILS — Control Panel access.",
        "FLASK_DEBUG — unset in production; set to true only for local development.",
    ])
    y -= 6
    box(c, M, y, W - 2 * M, None, "Verifying a deploy", [
        "Space state and commit:  https://huggingface.co/api/spaces/aditya-rAj19/FaceSWAP/runtime",
        "Expect \"stage\":\"RUNNING\" with sha matching the pushed commit, then probe / and /api/locations.",
    ], fill=(0.94, 0.98, 0.97))


# ── page 6 — behaviour + gotchas ─────────────────────────────────────────────
def page_notes(c):
    y = header(c, "Behaviour and known limits", "Things that are easy to misread from the code")

    y = box(c, M, y, W - 2 * M, None, "CPU and GPU", [
        "Both paths run the same pipeline; only cost and working resolution differ.",
        "CPU   1024px working resolution; RealESRGAN on the head crop (~6 min).",
        "GPU   1536px; RealESRGAN available for the full frame; local HairFastGAN possible.",
        "Full-frame super-resolution on CPU was measured at about 20 minutes, which is why",
        "it is confined to the head.",
    ]) - 14

    y = box(c, M, y, W - 2 * M, None, "Hair transfer", [
        "Real hair transfer requires HairFastGAN. On the Space it connects to the public",
        "AIRI-Institute/HairFastGAN ZeroGPU Space anonymously and hits its quota, so it fails",
        "and the target's own hair is kept.",
        "",
        "To enable it: set HF_TOKEN on the Space, or point HAIRFAST_SPACE at your own GPU",
        "Space. There is deliberately no fallback — warping hair in directly looked pasted.",
    ], fill=(0.98, 0.96, 0.90), accent=AMBER) - 14

    y = box(c, M, y, W - 2 * M, None, "Opt-in stages, and why", [
        "Head swap    changes far more of the photo than a face swap; exposed as a toggle.",
        "Hair         only runs when HairFastGAN returns a portrait.",
        "Glasses      warped spectacles onto the face as a misaligned, translucent shape.",
        "Each is a form field on /api/swap, so any of them can be turned back on per request.",
    ]) - 14

    y = box(c, M, y, W - 2 * M, None, "Quality metrics", [
        "Reported per swap: alignment, blend, delta-E and naturalness. They are computed",
        "BEFORE the harmonise stage — that stage adds grain deliberately, and scoring after it",
        "penalised the pipeline for its own noise, understating every result.",
    ]) - 14

    box(c, M, y, W - 2 * M, None, "Privacy", [
        "Uploads are decoded in memory and the result is returned as a data URI; nothing is",
        "written to disk. Finished jobs, which hold the images, expire after 20 minutes.",
    ])


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "WORKFLOW.pdf")

    pages = [page_overview, page_architecture, page_request,
             page_pipeline, page_deploy, page_notes]

    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("DeepFace Studio — project workflow")
    c.setAuthor("Aditya Raj")
    for i, page in enumerate(pages, 1):
        page(c)
        footer(c, i, len(pages))
        c.showPage()
    c.save()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
