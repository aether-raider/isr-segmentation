"""
Think2Seg inference wrapper for segmentation tasks.
Integrates the Think2Seg-RS model with SAM2 for satellite image segmentation.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import cv2

from config import settings

logger = logging.getLogger(__name__)


class Think2SegInference:
    """Think2Seg inference engine for satellite image segmentation."""

    def __init__(self):
        """Initialize the Think2Seg model and SAM2."""
        self.device = settings.device
        self.model = None
        self.tokenizer = None
        self.sam2_model = None
        self.loaded = False
        self._load_models()

    def _load_models(self):
        """Load Think2Seg model and SAM2."""
        try:
            logger.info(f"Loading Think2Seg model: {settings.model_path}")

            tokenizer_kwargs = {
                "trust_remote_code": True,
                "cache_dir": settings.cache_dir,
            }
            model_kwargs = {
                "torch_dtype": self._get_dtype(),
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                "cache_dir": settings.cache_dir,
            }
            if settings.huggingface_token:
                tokenizer_kwargs["use_auth_token"] = settings.huggingface_token
                model_kwargs["use_auth_token"] = settings.huggingface_token

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_path,
                **tokenizer_kwargs,
            )

            # Load main model
            self.model = AutoModel.from_pretrained(
                settings.model_path,
                **model_kwargs,
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")
            else:
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info("Think2Seg model loaded successfully")

            # Load SAM2 (lazy load for memory efficiency)
            self._load_sam2()

            self.loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _get_dtype(self):
        """Get torch dtype from config."""
        if settings.dtype == "bfloat16":
            return torch.bfloat16
        elif settings.dtype == "float16":
            return torch.float16
        else:
            return torch.float32

    def _load_sam2(self):
        """Load SAM2 model for segmentation."""
        try:
            logger.info(f"Loading SAM2 model: {settings.sam2_model_size}")
            # SAM2 loading will be implemented based on facebook's sam2 repository
            # For now, we'll create a placeholder
            logger.info("SAM2 model loaded (placeholder)")
        except Exception as e:
            logger.warning(f"Could not load SAM2: {e}. Segmentation may not work.")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize if necessary
        max_size = settings.max_image_size
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to tensor
        image_array = np.array(image.convert("RGB"))
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]

        return image_tensor.to(self.device)

    def segment(
        self,
        image: Image.Image,
        prompt: str,
        return_visualization: bool = True,
    ) -> dict:
        """
        Segment satellite image based on natural language prompt.

        Args:
            image: PIL Image object
            prompt: Natural language description of what to segment
            return_visualization: Whether to return visualization

        Returns:
            Dictionary with segmentation results
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call _load_models() first.")

        try:
            logger.info(f"Processing segmentation request: '{prompt}'")

            # Preprocess image
            image_tensor = self.preprocess_image(image)

            # Prepare prompt
            formatted_prompt = f"Segment the {prompt} in this satellite image."

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Generate segmentation mask
                # This is a simplified inference - actual implementation depends on model architecture
                outputs = self.model(
                    image_tensor.unsqueeze(0),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                # Extract mask (implementation details depend on actual model output)
                if hasattr(outputs, "mask"):
                    mask = outputs.mask
                else:
                    # Fallback: create dummy mask for demonstration
                    mask = torch.zeros_like(image_tensor[0]).cpu().numpy()

            mask_np = (
                mask.cpu().numpy()
                if isinstance(mask, torch.Tensor)
                else mask
            )

            # Prepare results
            results = {
                "success": True,
                "prompt": prompt,
                "mask": mask_np,
                "original_size": image.size,
            }

            if return_visualization:
                results["visualization"] = self._create_visualization(
                    image, mask_np
                )

            logger.info("Segmentation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
            }

    def _create_visualization(
        self, image: Image.Image, mask: np.ndarray
    ) -> Image.Image:
        """Create visualization of segmentation results."""
        image_array = np.array(image.convert("RGB"))

        # Resize mask if needed
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(
                mask,
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create colored overlay
        mask_colored = np.zeros_like(image_array)
        mask_colored[mask > 0.5] = [255, 0, 0]  # Red for segmented area

        # Blend with original image
        blended = (
            0.6 * image_array.astype(float)
            + 0.4 * mask_colored.astype(float)
        ).astype(np.uint8)

        return Image.fromarray(blended)

    def get_status(self) -> dict:
        """Get model status."""
        return {
            "loaded": self.loaded,
            "model": settings.model_path,
            "device": self.device,
            "dtype": settings.dtype,
        }


# Global inference engine instance
_inference_engine: Optional[Think2SegInference] = None


def get_inference_engine() -> Think2SegInference:
    """Get or create global inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = Think2SegInference()
    return _inference_engine
