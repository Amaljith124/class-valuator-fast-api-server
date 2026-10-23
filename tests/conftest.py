import os

os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost:5432/test")
os.environ.setdefault("R2_ACCOUNT_ID", "test-account")
os.environ.setdefault("R2_ACCESS_KEY_ID", "test-access-key")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "test-secret-key")
os.environ.setdefault("R2_BUCKET_NAME", "test-bucket")
os.environ.setdefault("R2_ENDPOINT", "https://test-account.r2.cloudflarestorage.com")
os.environ.setdefault("GEMINI_API_KEY", "test-api-key")
os.environ.setdefault("INTERNAL_API_TOKEN", "test-internal-token")
