import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Model Configuration
    model_name: str = "Think2Seg-RS-7B"  # or "Think2Seg-RS-3B"
    model_path: Optional[str] = None  # HuggingFace model ID or local path
    huggingface_token: Optional[str] = None  # Optional Hugging Face access token for gated repos
    device: str = "cuda"  # or "cpu"
    dtype: str = "bfloat16"  # or "float32"

    # SAM2 Configuration
    sam2_model_size: str = "base_plus"  # tiny, small, base_plus, large
    sam2_device: str = "cuda"

    # Inference Settings
    max_image_size: int = 1024  # Maximum image dimension
    inference_timeout: int = 300  # seconds
    batch_size: int = 1

    # Model Caching
    use_cache: bool = True
    cache_dir: str = ".model_cache"

    # VLLM Configuration (for faster inference)
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1

    class Config:
        env_file = ".env"
        case_sensitive = False

    def model_post_init(self, __context):
        """Post-initialization validation."""
        if self.model_path is None:
            if "3B" in self.model_name:
                self.model_path = "RicardoString/Think2Seg-RS-3B"
            else:
                self.model_path = "RicardoString/Think2Seg-RS-7B"


settings = Settings()
