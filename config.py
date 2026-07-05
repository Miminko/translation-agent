from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    whisper_model: str = "whisper-1"
    translation_model: str = "gpt-4.1-mini"
    data_dir: str = "./data"
    whisper_mode: str = "always"  # always | fallback_only

    class Config:
        env_file = ".env"

settings = Settings()  # fails fast if OPENAI_API_KEY missing
