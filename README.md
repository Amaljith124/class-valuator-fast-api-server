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

The first transcription loads the Whisper model, which faster-whisper normally downloads from Hugging Face. Allow network access and extra time on the first run.

## Clone and setup order

This service is the internal AI pipeline of a three-part system. Clone and start the parts in this order:

1. **PostgreSQL** and a private **Cloudflare R2** bucket. The Node server owns the schema, so apply its scripts first from the Node repo's `database/scripts/`.
2. **FastAPI analysis service** (this repo): clone it, then follow "Setup" above and start it with the `uvicorn` command given there.

```sh
   git clone <your-fastapi-repo-url>
   cd <your-fastapi-repo-folder>
```

   Confirm `GET /health` returns `{"success": true, "data": {"status": "ok"}}`.
3. **Node API**: clone [class-valuator-node-server](https://github.com/Amaljith124/class-valuator-node-server.git), set `FASTAPI_BASE_URL` to this service (for example `http://127.0.0.1:8000`) and `FASTAPI_INTERNAL_TOKEN` to the same value as this service's `INTERNAL_API_TOKEN`.
4. **React UI**: clone [class-valuator-react-site](https://github.com/Amaljith124/class-valuator-react-site.git) and point it at the Node API.

Settings that must match across services: `INTERNAL_API_TOKEN` here equals `FASTAPI_INTERNAL_TOKEN` in Node; the `R2_*` settings and the database must be the same ones Node uses.

## Pipeline boundaries

Transcription uses local faster-whisper. Teacher/student labels use transcript context, not acoustic diarization. Video analysis samples at most MAX_VIDEO_FRAMES frames at FRAME_INTERVAL_SECONDS; it does not inspect continuous video. Audio scores are signal-derived heuristics and are not a human intelligibility test. Teaching scores are deterministic weighted means over assessable criteria with renormalized weights; audio/video remain separate.

The user manually applies schema setup and updates from the Node server repo's `database/scripts/` when required. FastAPI never runs them.

## Tests and temporary files

Install pytest (`python -m pip install pytest`) and run `python -m pytest tests` from this directory.

Temporary media is written to `./tmp` (`TEMP_DIR`) while a job runs. It is safe to clear that folder when the service is stopped.
