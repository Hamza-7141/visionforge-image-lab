# VisionForge — Image Learning Lab

Train and compare three real image classifiers from labeled examples: Logistic Regression, Random Forest, and a small TensorFlow CNN. A student-friendly FastAPI application with a responsive HTML/CSS/JavaScript interface.

## What it can do

- Upload examples into 2–6 named classes, with at least 3 images each (10+ recommended).
- Train all three models and view measured held-out validation accuracy and CNN learning curves.
- Predict from an uploaded image or an explicitly enabled camera snapshot.
- Download Logistic Regression and Random Forest as ONNX, with labels and preprocessing instructions. CNN export is not supported.
- Switch light/dark appearance for the current page and clear the temporary dataset and models.

## Privacy

No signup, analytics, external chart CDN, database, cookies, browser storage, or persistent image/model files. Images are transmitted to this site's own server only when Upload or Capture & Predict is clicked; they are decoded into in-memory arrays. Only use non-sensitive images. Camera access is opt-in and can be stopped. Each page has a separate random identifier in memory; no login or account exists.

Sessions expire after 20 minutes of inactivity (cleanup runs every 30 seconds) and disappear on server restart. Refresh starts a new session; Clear removes the current session immediately unless training is active. The hosting provider may retain normal infrastructure logs. Cloud scaling can reset or lose in-memory sessions; this is a small educational demo, not a durable training service.

## Local setup

Python 3.13 is required by the tested dependencies. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\run.cmd
```

Open http://127.0.0.1:8001. Stop with Ctrl+C. Test dependencies additionally require httpx:

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Structure

- `backend/app.py`: API, per-visitor memory, resource limits and static frontend serving.
- `backend/utils.py`: image validation, resizing, normalization and class-balanced splits.
- `backend/trainers.py`: real classifiers, metrics and memory-only ONNX export.
- `backend/inference.py`: predictions from the current session's models.
- `frontend/`: existing interface, bundled Chart.js, opt-in camera and same-origin API calls.
- `Dockerfile.vercel`: CPU container for Vercel's Container Runtime (beta), no additional data service.

## Assumptions and limits

Images become 64×64 RGB float32 arrays in [0,1]. A deterministic per-class 80/20 split rounds validation counts upward, with at least one validation example per class. All models use the same split. Four CNN epochs, 50 random-forest trees, one active training job per container, at most 12 temporary sessions, 120 images per session, 512 KB per upload, and 16 megapixels per decoded image. Multipart request sizes are bounded below the parser's disk-spooling threshold.

Validation accuracy from a tiny user-selected dataset is unreliable and can be inflated by similar examples. Confidence scores are not calibrated certainty. This is not a medical, identity, security or production recognition tool. CPU training and cold starts can take time. No GPU is promised. Uploads invalidate existing models, so retraining uses the current dataset.

The original Teachable-Machine-style hackathon interface is preserved with targeted privacy and deployment fixes. Chart.js 4.5.1 is bundled locally under its MIT license.
