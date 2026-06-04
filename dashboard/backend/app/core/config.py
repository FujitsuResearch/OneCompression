from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./onecomp.db"
    redis_url: str = "redis://127.0.0.1:6379/0"

    quantized_dir: str = "tmp/quantized"

    mock_quantization: bool = False
    device: Literal["cpu", "mps", "cuda"] = "cpu"

    worker_host: str = "localhost"
    vllm_port: int = 8090
    vllm_python: str = ".venv/bin/python"

    chat_timeout: int = 900

    model_config = {"env_prefix": "ONECOMP_"}


settings = Settings()
