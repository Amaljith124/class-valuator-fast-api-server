# Humaniti FastAPI analysis server

Python 3.11+ internal service. The Node server is the only public API and owns the schema and all SQL scripts. This service uses the existing evaluation table only and contains no DDL, migrations, schema initialization, or .sql files.

## Setup

Create and activate a virtual environment, install requirements, copy .env.example to .env, and configure DATABASE_URL, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT, GEMINI_API_KEY, and INTERNAL_API_TOKEN (the latter must match Node). Use least-privilege R2 credentials and keep the bucket private. GEMINI_MODEL selects the model used for text and image analysis.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install FFmpeg separately: macOS brew install ffmpeg; Debian/Ubuntu sudo apt-get install ffmpeg.

Run `python -m uvicorn app.main:app --reload --port 8000` from this directory while the virtual environment is active. GET /health returns service health. Node calls POST /api/v1/analyze with {"object_key":"recordings/example.mp4","evaluation_id":"..."} and x-internal-token. Four protected test routes are available under /api/v1/test/.

## Pipeline boundaries

Transcription uses local faster-whisper. Teacher/student labels use transcript context, not acoustic diarization. Video analysis samples at most MAX_VIDEO_FRAMES frames at FRAME_INTERVAL_SECONDS; it does not inspect continuous video. Audio scores are signal-derived heuristics and are not a human intelligibility test. Teaching scores are deterministic weighted means over assessable criteria with renormalized weights; audio/video remain separate.

The user manually applies schema setup and updates from ../node-server/database/scripts/ when required. FastAPI never runs them.
