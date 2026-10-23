from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Humaniti Video Analysis"
    app_env: str = "development"
    database_url: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    r2_endpoint: str
    r2_region: str = "auto"
    gemini_api_key: str
    gemini_model: str = "gemini-3.5-flash"
    gemini_light_model: str = "gemini-3.5-flash-lite"
    internal_api_token: str
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_concurrency: int = 1
    whisper_batch_size: int = 8
    whisper_beam_size: int = 1
    whisper_language: str | None = None
    whisper_cpu_threads: int = 0
    llm_batch_size: int = 120
    llm_concurrency: int = 3
    gemini_timeout_ms: int = 120000
    transcript_checkpoint_enabled: bool = True
    temp_dir: str = "./tmp"
    max_video_frames: int = 12
    frame_interval_seconds: int = 30
    max_media_mb: int = 500
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

settings = Settings()
