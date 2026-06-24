from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config: ConfigDict = ConfigDict(
        protected_namespaces=("settings_",),
        env_file=".env",
        case_sensitive=False,
        extra="allow",
    )

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8123
    debug: bool = False
    frontend_port: int = 3123
    vite_port: int = 5173
    cors_origins: list[str] = Field(default_factory=list)

    # Model Configuration
    model_name: str = "Think2Seg-RS-7B"  # or "Think2Seg-RS-3B"
    model_path: Optional[str] = None  # HuggingFace model ID or local path
    hf_token: Optional[str] = Field(
        default=None,
        env="HF_TOKEN",
        description="Optional Hugging Face access token for gated repos",
    )
    device: str = "cuda"  # or "cpu"
    model_device_map: str = "auto"  # auto shards the VLM across visible GPUs
    dtype: str = "bfloat16"  # or "float32"

    # SAM2 Configuration
    sam2_model_size: str = "base_plus"  # tiny, small, base_plus, large
    sam2_device: str = "cuda"
    sam2_root: str = "/opt/sam2"
    sam2_checkpoint: Optional[str] = None
    sam2_config: Optional[str] = None

    # Inference Settings
    max_image_size: int = 1024  # Maximum image dimension
    inference_timeout: Optional[int] = None
    batch_size: int = 1
    max_new_tokens: int = 1024
    clear_cuda_cache_after_run: bool = True
    enable_tf32: bool = True
    output_dir: str = "outputs"
    aether_min_component_area: int = 50
    aether_dedup_iou_threshold: float = 0.55
    aether_dedup_containment_threshold: float = 0.85
    aether_classification_smoothing: float = 0.01

    # Ultra-Sim callback handoff
    ultra_sim_callback_base_url: Optional[str] = None
    ultra_sim_callback_token: Optional[str] = None
    ultra_sim_callback_timeout: Optional[int] = None

    # Model Caching
    use_cache: bool = True
    cache_dir: str = ".model_cache"

    # VLLM Configuration (for faster inference)
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1

    def model_post_init(self, __context):
        """Post-initialization validation."""
        if self.model_path is None:
            if "3B" in self.model_name:
                self.model_path = "RicardoString/Think2Seg-RS-3B"
            else:
                self.model_path = "RicardoString/Think2Seg-RS-7B"

        if not self.cors_origins:
            self.cors_origins = [
                f"http://localhost:{self.frontend_port}",
                f"http://127.0.0.1:{self.frontend_port}",
                f"http://localhost:{self.vite_port}",
                f"http://127.0.0.1:{self.vite_port}",
            ]


settings = Settings()
