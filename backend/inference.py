"""Think2Seg inference wrapper for satellite-image segmentation."""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import cv2

from config import settings

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]

SYSTEM_PROMPT = (
    "You are a remote sensing analysis assistant. Your task is to generate "
    "spatial prompts for the Segment Anything Model (SAM) based on a user's "
    "request."
)
USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
1. `bbox_2d`: a tight bounding box as [x1, y1, x2, y2].
2. `positive_points`: exactly two points inside the target.
Output your thinking process in <think> </think> tags.
Output the final answer in <answer> </answer> tags as a valid JSON array inside a JSON code fence. If no targets are found, output [].
Every object must include the `bbox_2d` key. Do not output bare coordinate arrays or omit JSON keys.
Example:
<think>thinking process here</think>
<answer>```json
[
  {{"bbox_2d": [310, 360, 567, 586], "positive_points": [[434, 474], [450, 460]]}},
  {{"bbox_2d": [10, 200, 100, 320], "positive_points": [[50, 250], [90, 300]]}}
]
```</answer>"""

DEFAULT_SEGMENTATION_OPTIONS = {
    "sam_mask_threshold": 0.0,
    "sam_multimask_output": False,
    "mask_min_area": 0,
    "mask_cleanup_px": 0,
    "mask_expand_px": 0,
    "refinement_passes": 0,
    "refinement_mode": "intersection",
}


class Think2SegInference:
    """Think2Seg inference engine for satellite image segmentation."""

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        """Initialize the Think2Seg model and SAM2."""
        self.device = settings.device.lower()
        self.input_device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.sam2_predictor = None
        self.sam2_error = None
        self.loaded = False
        self._load_models(progress_callback=progress_callback)

    def _load_models(self, progress_callback: Optional[ProgressCallback] = None):
        """Load Think2Seg model and SAM2."""
        try:
            logger.info(f"Loading Think2Seg model: {settings.model_path}")
            self._emit_progress(progress_callback, 2, "Configuring CUDA runtime")

            tokenizer_kwargs = {
                "trust_remote_code": True,
                "cache_dir": settings.cache_dir,
            }
            model_device_map = None
            if self.device.startswith("cuda"):
                model_device_map = settings.model_device_map or "auto"
                if model_device_map.lower() in {"none", "false", "off"}:
                    model_device_map = None
            model_kwargs = {
                "torch_dtype": self._get_dtype(),
                "device_map": model_device_map,
                "trust_remote_code": True,
                "cache_dir": settings.cache_dir,
            }
            max_memory = self._get_model_max_memory(model_device_map)
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            if settings.hf_token:
                tokenizer_kwargs["token"] = settings.hf_token
                model_kwargs["token"] = settings.hf_token

            self._configure_cuda_runtime()

            # Load processor and VLM. Think2Seg-RS is a Qwen2.5-VL generation
            # model that predicts structured SAM2 prompts, not masks directly.
            self._emit_progress(progress_callback, 8, "Loading Qwen processor")
            self.processor = AutoProcessor.from_pretrained(
                settings.model_path,
                **tokenizer_kwargs,
            )

            self._emit_progress(progress_callback, 14, "Loading Qwen vision-language model")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                settings.model_path,
                **model_kwargs,
            )

            if getattr(self.model, "hf_device_map", None):
                logger.info(
                    "Think2Seg model dispatched with Accelerate device map; "
                    "skipping manual .to(%s)",
                    self.device,
                )
            else:
                self.model = self.model.to(self.device)

            self.model.eval()
            self.input_device = self._get_input_device()
            self._emit_progress(progress_callback, 34, "Qwen model loaded")

            logger.info("Think2Seg model loaded successfully")

            self._emit_progress(progress_callback, 38, "Loading SAM2 model")
            self._load_sam2()

            self.loaded = True
            if self.sam2_predictor is None:
                logger.warning("SAM2 was not loaded: %s", self.sam2_error)
            else:
                logger.info("All models loaded successfully")
            self._emit_progress(progress_callback, 45, "Models ready")

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

    def _configure_cuda_runtime(self) -> None:
        """Enable CUDA runtime optimizations that are safe for inference."""
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return

        if settings.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                logger.debug("Could not set float32 matmul precision", exc_info=True)

    def _get_model_max_memory(self, model_device_map) -> Optional[dict]:
        """Let Accelerate use a configurable share of currently free GPU memory."""
        if (
            not model_device_map
            or not self.device.startswith("cuda")
            or not torch.cuda.is_available()
        ):
            return None

        utilization = min(
            max(float(settings.model_gpu_memory_utilization or 0.95), 0.1),
            0.99,
        )
        explicit_gb = settings.model_max_memory_gb
        max_memory = {}
        for index in range(torch.cuda.device_count()):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                budget_bytes = int(free_bytes * utilization)
                if explicit_gb:
                    budget_bytes = min(
                        budget_bytes,
                        int(float(explicit_gb) * 1024**3),
                    )
                max_memory[index] = f"{max(1, budget_bytes // 1024**2)}MiB"
            except Exception:
                logger.debug("Could not calculate max memory for GPU %s", index, exc_info=True)

        if max_memory:
            logger.info("Model max_memory map: %s", max_memory)
        return max_memory or None

    def _get_input_device(self) -> torch.device:
        """Return the device where request tensors should enter the model."""
        device_map = getattr(self.model, "hf_device_map", None) or {}
        for mapped_device in device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, torch.device):
                if mapped_device.type not in {"cpu", "meta"}:
                    return mapped_device
                continue
            if isinstance(mapped_device, str) and mapped_device.isdigit():
                return torch.device(f"cuda:{mapped_device}")
            if mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")

        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass

        return torch.device("cpu")

    def _load_sam2(self):
        """Load SAM2 model for segmentation."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_root = Path(settings.sam2_root).expanduser()
            model_cfg, checkpoint = self._get_sam2_paths(sam2_root)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found at {checkpoint}. "
                    "Install SAM2 checkpoints or set SAM2_ROOT/SAM2_CHECKPOINT."
                )

            logger.info(
                "Loading SAM2 %s model from %s",
                settings.sam2_model_size,
                checkpoint,
            )
            sam2_model = build_sam2(
                model_cfg,
                str(checkpoint),
                device=settings.sam2_device,
            )
            for param in sam2_model.parameters():
                param.requires_grad = False
            sam2_model.eval()

            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            self.sam2_error = None
            logger.info("SAM2 model loaded successfully")
        except Exception as e:
            self.sam2_predictor = None
            self.sam2_error = str(e)
            logger.warning("Could not load SAM2: %s", e)

    def _get_sam2_paths(self, sam2_root: Path) -> tuple[str, Path]:
        """Return SAM2 config name and checkpoint path for the configured size."""
        defaults = {
            "large": (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "checkpoints/sam2.1_hiera_large.pt",
            ),
            "base_plus": (
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "checkpoints/sam2.1_hiera_base_plus.pt",
            ),
            "small": (
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                "checkpoints/sam2.1_hiera_small.pt",
            ),
            "tiny": (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                "checkpoints/sam2.1_hiera_tiny.pt",
            ),
        }
        try:
            model_cfg, checkpoint = defaults[settings.sam2_model_size]
        except KeyError as exc:
            raise ValueError(
                f"Invalid SAM2_MODEL_SIZE '{settings.sam2_model_size}'. "
                f"Expected one of: {', '.join(defaults)}"
            ) from exc

        if settings.sam2_config:
            model_cfg = settings.sam2_config
        if settings.sam2_checkpoint:
            checkpoint_path = Path(settings.sam2_checkpoint).expanduser()
        else:
            checkpoint_path = sam2_root / checkpoint

        return model_cfg, checkpoint_path

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for model input."""
        processed = image.convert("RGB").copy()
        max_size = settings.max_image_size
        if max(processed.size) > max_size:
            processed.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return processed

    def _normalize_options(self, options: Optional[dict]) -> dict:
        normalized = dict(DEFAULT_SEGMENTATION_OPTIONS)
        if options:
            normalized.update({
                key: value
                for key, value in options.items()
                if key in normalized and value is not None
            })

        normalized["sam_mask_threshold"] = self._clamp_float(
            normalized["sam_mask_threshold"],
            -2.0,
            2.0,
            DEFAULT_SEGMENTATION_OPTIONS["sam_mask_threshold"],
        )
        normalized["sam_multimask_output"] = bool(normalized["sam_multimask_output"])
        normalized["mask_min_area"] = self._clamp_int(
            normalized["mask_min_area"],
            0,
            2_000_000,
            DEFAULT_SEGMENTATION_OPTIONS["mask_min_area"],
        )
        normalized["mask_cleanup_px"] = self._clamp_int(
            normalized["mask_cleanup_px"],
            0,
            31,
            DEFAULT_SEGMENTATION_OPTIONS["mask_cleanup_px"],
        )
        normalized["mask_expand_px"] = self._clamp_int(
            normalized["mask_expand_px"],
            -31,
            31,
            DEFAULT_SEGMENTATION_OPTIONS["mask_expand_px"],
        )
        normalized["refinement_passes"] = self._clamp_int(
            normalized["refinement_passes"],
            0,
            3,
            DEFAULT_SEGMENTATION_OPTIONS["refinement_passes"],
        )
        if normalized["refinement_mode"] not in {"intersection", "union", "replace"}:
            normalized["refinement_mode"] = DEFAULT_SEGMENTATION_OPTIONS[
                "refinement_mode"
            ]
        return normalized

    @staticmethod
    def _clamp_float(value, minimum: float, maximum: float, fallback: float) -> float:
        try:
            return min(max(float(value), minimum), maximum)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _clamp_int(value, minimum: int, maximum: int, fallback: int) -> int:
        try:
            return min(max(int(value), minimum), maximum)
        except (TypeError, ValueError):
            return fallback

    def segment(
        self,
        image: Image.Image,
        prompt: str,
        return_visualization: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
        options: Optional[dict] = None,
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
            options = self._normalize_options(options)
            logger.info(f"Processing segmentation request: '{prompt}'")
            self._emit_progress(progress_callback, 5, "Checking model readiness")
            if self.sam2_predictor is None:
                raise RuntimeError(f"SAM2 is not loaded: {self.sam2_error}")

            self._emit_progress(progress_callback, 12, "Preparing image")
            processed_image = self.preprocess_image(image)

            mask_np = None
            all_instances = []
            model_outputs = []
            pass_outputs = []
            target_outputs = []
            refinement_image = processed_image
            total_passes = options["refinement_passes"] + 1

            for pass_index in range(total_passes):
                pass_number = pass_index + 1
                pass_start = 16 + int((pass_index / total_passes) * 74)
                pass_end = 16 + int((pass_number / total_passes) * 74)
                self._emit_progress(
                    progress_callback,
                    pass_start,
                    f"Pass {pass_number}/{total_passes}: generating spatial prompts",
                )
                model_output = self._generate_sam_prompt(refinement_image, prompt)
                model_outputs.append(model_output)
                self._emit_progress(
                    progress_callback,
                    min(pass_start + 10, pass_end),
                    f"Pass {pass_number}/{total_passes}: parsing target prompts",
                )
                instances, parse_error = self._parse_sam_prompts(model_output)
                if instances is None:
                    raise ValueError(
                        "Could not parse SAM prompts from model output: "
                        f"{parse_error}. Output was: {model_output}"
                    )
                all_instances.extend(instances)

                self._emit_progress(
                    progress_callback,
                    min(pass_start + 18, pass_end),
                    (
                        f"Pass {pass_number}/{total_passes}: running SAM2 on "
                        f"{len(instances)} target prompt(s)"
                    ),
                )
                pass_mask, pass_targets = self._predict_sam_mask(
                    refinement_image,
                    instances,
                    progress_callback=progress_callback,
                    options=options,
                )
                pass_target_indices = []
                if return_visualization:
                    for pass_target in pass_targets:
                        target_number = len(target_outputs) + 1
                        target_mask = self._postprocess_mask(
                            pass_target["mask"],
                            options,
                        )
                        target_outputs.append({
                            "target_number": target_number,
                            "pass_number": pass_number,
                            "target_in_pass": pass_target["target_in_pass"],
                            "score": pass_target.get("score"),
                            "sam_prompt": pass_target.get("sam_prompt"),
                            "mask": target_mask.copy(),
                            "visualization": self._create_visualization(
                                processed_image,
                                target_mask,
                            ),
                            "segmented_image": self._create_segmented_image(
                                processed_image,
                                target_mask,
                            ),
                        })
                        pass_target_indices.append(target_number)

                mask_np = self._combine_refinement_mask(
                    mask_np,
                    pass_mask,
                    options["refinement_mode"],
                )
                mask_np = self._postprocess_mask(mask_np, options)
                if return_visualization:
                    pass_outputs.append({
                        "pass_number": pass_number,
                        "total_passes": total_passes,
                        "prompt_count": len(instances),
                        "model_output": model_output,
                        "sam_prompts": instances,
                        "target_numbers": pass_target_indices,
                        "mask": mask_np.copy(),
                        "visualization": self._create_visualization(
                            processed_image,
                            mask_np,
                        ),
                        "segmented_image": self._create_segmented_image(
                            processed_image,
                            mask_np,
                        ),
                    })
                refinement_image = self._create_refinement_image(
                    processed_image,
                    mask_np,
                )
                self._emit_progress(
                    progress_callback,
                    pass_end,
                    f"Pass {pass_number}/{total_passes}: mask refined",
                )

            results = {
                "success": True,
                "prompt": prompt,
                "mask": mask_np,
                "original_size": processed_image.size,
                "uploaded_size": image.size,
                "model_output": model_outputs[-1],
                "model_outputs": model_outputs,
                "sam_prompts": all_instances,
                "options": options,
                "refinement_passes_completed": total_passes,
            }

            if return_visualization:
                self._emit_progress(progress_callback, 94, "Rendering visualization")
                results["pass_outputs"] = pass_outputs
                results["target_outputs"] = target_outputs
                results["visualization"] = self._create_visualization(
                    processed_image,
                    mask_np,
                )
                results["segmented_image"] = self._create_segmented_image(
                    processed_image,
                    mask_np,
                )

            self._emit_progress(progress_callback, 98, "Packaging result")
            logger.info("Segmentation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
            }
        finally:
            self._emit_progress(progress_callback, 99, "Releasing temporary GPU cache")
            self.clear_runtime_cache()

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[ProgressCallback],
        progress: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(progress, message)
        except Exception:
            logger.exception("Progress callback failed")

    def _build_messages(self, image: Image.Image, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": USER_PROMPT.format(Question=prompt),
                    },
                ],
            },
        ]

    def _generate_sam_prompt(self, image: Image.Image, prompt: str) -> str:
        """Generate structured SAM2 prompts from the Qwen2.5-VL prompter."""
        messages = self._build_messages(image, prompt)

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

        inputs = self._move_inputs_to_device(inputs)
        generation_kwargs = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = getattr(
            getattr(self.processor, "tokenizer", None),
            "pad_token_id",
            None,
        )
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        completion_ids = generated_ids[:, prompt_length:]
        return self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _move_inputs_to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.input_device)
        return {
            key: value.to(self.input_device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _parse_sam_prompts(
        self,
        model_output: str,
    ) -> tuple[Optional[list[dict]], Optional[str]]:
        """Extract and validate Think2Seg's JSON SAM prompt list."""
        content = model_output
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()

        fence_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()

        if "[" in content and "]" in content:
            content = content[content.find("["): content.rfind("]") + 1]

        parse_errors = []
        for candidate in (content, self._repair_sam_json(content)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
        else:
            parsed = self._parse_sam_prompt_fragments(content)
            if parsed is None:
                return None, "JSON parsing error: " + "; ".join(parse_errors)

        if not isinstance(parsed, list):
            return None, "Output must be a JSON array"

        normalized = []
        for index, instance in enumerate(parsed):
            if not isinstance(instance, dict):
                return None, f"Instance #{index + 1} must be a JSON object"
            if "bbox_2d" not in instance and "positive_points" not in instance:
                return None, (
                    f"Instance #{index + 1} must contain bbox_2d or positive_points"
                )
            normalized.append(instance)

        return normalized, None

    def _repair_sam_json(self, content: str) -> Optional[str]:
        """Repair common Qwen JSON omissions such as missing bbox_2d keys."""
        if not content:
            return None

        repaired = content.strip()
        repaired = re.sub(
            r'\{\s*"(-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*'
            r'-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?)\]\s*,\s*"positive_points"',
            r'{"bbox_2d": [\1], "positive_points"',
            repaired,
        )
        repaired = re.sub(
            r"\{\s*\[(-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*"
            r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?)\]\s*,",
            r'{"bbox_2d": [\1],',
            repaired,
        )
        repaired = re.sub(r",\s*]", "]", repaired)
        repaired = re.sub(r",\s*}", "}", repaired)
        return repaired if repaired != content else None

    def _parse_sam_prompt_fragments(self, content: str) -> Optional[list[dict]]:
        """Fallback parser for malformed object fragments from the VLM."""
        instances = []
        object_fragments = re.findall(r"\{[^{}]*\}", content, flags=re.DOTALL)
        for fragment in object_fragments:
            instance = {}
            bbox = self._extract_bbox(fragment)
            if bbox is not None:
                instance["bbox_2d"] = bbox

            positive_points = self._extract_points(fragment, "positive_points")
            if positive_points:
                instance["positive_points"] = positive_points

            negative_points = self._extract_points(fragment, "negative_points")
            if negative_points:
                instance["negative_points"] = negative_points

            if "bbox_2d" in instance or "positive_points" in instance:
                instances.append(instance)

        return instances or None

    @staticmethod
    def _extract_bbox(fragment: str) -> Optional[list[float]]:
        bbox_match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', fragment)
        if bbox_match:
            source = bbox_match.group(1)
        else:
            source = fragment.split("positive_points", 1)[0]

        numbers = re.findall(r"-?\d+(?:\.\d+)?", source)
        if len(numbers) < 4:
            return None
        return [float(value) for value in numbers[:4]]

    @staticmethod
    def _extract_points(fragment: str, key: str) -> list[list[float]]:
        if key not in fragment:
            return []
        section = fragment.split(key, 1)[1]
        pairs = re.findall(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            section,
        )
        return [[float(x), float(y)] for x, y in pairs]

    def _predict_sam_mask(
        self,
        image: Image.Image,
        instances: list[dict],
        progress_callback: Optional[ProgressCallback] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Run SAM2 with Qwen-generated boxes/points and combine instance masks."""
        options = self._normalize_options(options)
        image_array = np.array(image.convert("RGB"))
        height, width = image_array.shape[:2]

        if not instances:
            self._emit_progress(progress_callback, 90, "No targets found")
            return np.zeros((height, width), dtype=np.uint8), []

        self._emit_progress(progress_callback, 66, "Computing SAM2 image embeddings")
        self.sam2_predictor.set_image(image_array)
        self._emit_progress(progress_callback, 78, "Predicting target masks")
        combined_mask = None
        target_masks = []

        with torch.inference_mode():
            total_instances = len(instances)
            for index, instance in enumerate(instances):
                bbox = self._normalize_box(instance.get("bbox_2d"), width, height)
                point_coords, point_labels = self._normalize_points(instance, width, height)
                if bbox is None and point_coords is None:
                    continue

                predict_kwargs = {
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "box": bbox,
                    "multimask_output": options["sam_multimask_output"],
                    "return_logits": True,
                }
                try:
                    masks, scores, _ = self.sam2_predictor.predict(**predict_kwargs)
                except TypeError:
                    predict_kwargs.pop("return_logits", None)
                    masks, scores, _ = self.sam2_predictor.predict(**predict_kwargs)
                if masks.shape[0] == 0:
                    continue

                best_mask_index = 0
                if options["sam_multimask_output"] and scores is not None:
                    best_mask_index = int(np.asarray(scores).argmax())

                raw_mask = np.asarray(masks[best_mask_index])
                if raw_mask.dtype == bool:
                    instance_mask = raw_mask
                else:
                    instance_mask = raw_mask > options["sam_mask_threshold"]
                score = None
                if scores is not None and len(scores) > best_mask_index:
                    try:
                        score = float(np.asarray(scores)[best_mask_index])
                    except (TypeError, ValueError):
                        score = None
                target_masks.append({
                    "target_in_pass": index + 1,
                    "score": score,
                    "sam_prompt": instance,
                    "raw_mask": raw_mask.astype(np.float32, copy=False),
                    "mask": instance_mask.astype(np.uint8),
                })
                combined_mask = (
                    instance_mask
                    if combined_mask is None
                    else np.logical_or(combined_mask, instance_mask)
                )
                self._emit_progress(
                    progress_callback,
                    78 + int(((index + 1) / total_instances) * 12),
                    f"Predicted mask {index + 1} of {total_instances}",
                )

        if combined_mask is None:
            raise RuntimeError("SAM2 did not produce a mask for the generated prompts")
        self._emit_progress(progress_callback, 92, "Combining masks")
        return self._postprocess_mask(combined_mask.astype(np.uint8), options), target_masks

    def _normalize_box(self, box, width: int, height: int):
        if box is None:
            return None
        if not isinstance(box, list) or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in box]
        except (TypeError, ValueError):
            return None

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x2, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        y2 = min(max(y2, 0), height - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _normalize_points(self, instance: dict, width: int, height: int):
        coords = []
        labels = []
        for key, label in (("positive_points", 1), ("negative_points", 0)):
            points = instance.get(key) or []
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                coords.append([
                    min(max(x, 0), width - 1),
                    min(max(y, 0), height - 1),
                ])
                labels.append(label)

        if not coords:
            return None, None
        return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _combine_refinement_mask(
        self,
        current_mask: Optional[np.ndarray],
        pass_mask: np.ndarray,
        refinement_mode: str,
    ) -> np.ndarray:
        if current_mask is None:
            return pass_mask.astype(np.uint8)
        if refinement_mode == "union":
            return np.logical_or(current_mask > 0, pass_mask > 0).astype(np.uint8)
        if refinement_mode == "replace":
            return pass_mask.astype(np.uint8)
        return np.logical_and(current_mask > 0, pass_mask > 0).astype(np.uint8)

    def _postprocess_mask(self, mask: np.ndarray, options: dict) -> np.ndarray:
        processed = (mask > 0).astype(np.uint8)

        cleanup_px = options["mask_cleanup_px"]
        if cleanup_px > 0:
            kernel_size = cleanup_px if cleanup_px % 2 == 1 else cleanup_px + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        expand_px = options["mask_expand_px"]
        if expand_px != 0:
            kernel_size = abs(expand_px) * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            if expand_px > 0:
                processed = cv2.dilate(processed, kernel, iterations=1)
            else:
                processed = cv2.erode(processed, kernel, iterations=1)

        min_area = options["mask_min_area"]
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                processed,
                connectivity=8,
            )
            kept = np.zeros_like(processed)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    kept[labels == label] = 1
            processed = kept

        return processed.astype(np.uint8)

    def _create_refinement_image(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        image_array = np.array(image.convert("RGB"))
        height, width = image_array.shape[:2]
        binary_mask = self._resize_mask(mask, width, height)
        masked = np.zeros_like(image_array)
        masked[binary_mask > 0] = image_array[binary_mask > 0]
        return Image.fromarray(masked)

    def clear_runtime_cache(self) -> None:
        """Release per-request image state and temporary CUDA allocator cache."""
        if not settings.clear_cuda_cache_after_run:
            return

        if self.sam2_predictor is not None:
            for method_name in ("reset_predictor", "reset_image"):
                reset_method = getattr(self.sam2_predictor, method_name, None)
                if callable(reset_method):
                    try:
                        reset_method()
                    except Exception:
                        logger.debug(
                            "Could not reset SAM2 predictor with %s",
                            method_name,
                            exc_info=True,
                        )
                    break

        gc.collect()
        if not torch.cuda.is_available():
            return

        for device_index in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_index):
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug(
                    "Could not clear CUDA cache on device %s",
                    device_index,
                    exc_info=True,
                )

        try:
            torch.cuda.ipc_collect()
        except Exception:
            logger.debug("Could not run CUDA IPC collection", exc_info=True)

    def _create_visualization(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """Create a high-contrast mask overlay without prompt boxes or points."""
        image_array = np.array(image.convert("RGB"))
        height, width = image_array.shape[:2]
        binary_mask = self._resize_mask(mask, width, height)

        mask_colored = np.zeros_like(image_array)
        mask_colored[binary_mask > 0] = [42, 214, 255]
        blended = (
            0.62 * image_array.astype(float)
            + 0.38 * mask_colored.astype(float)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            cv2.drawContours(blended, contours, -1, (0, 0, 0), 5)
            cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)

        return Image.fromarray(blended)

    def _create_segmented_image(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """Return only segmented pixels on a transparent background."""
        image_array = np.array(image.convert("RGBA"))
        height, width = image_array.shape[:2]
        binary_mask = self._resize_mask(mask, width, height)
        image_array[..., 3] = (binary_mask * 255).astype(np.uint8)
        return Image.fromarray(image_array)

    @staticmethod
    def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask > 0.5).astype(np.uint8)

    def get_status(self) -> dict:
        """Get model status."""
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(gpu_count)
        ]
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            hf_device_map = {
                key: str(value)
                for key, value in hf_device_map.items()
            }

        return {
            "loaded": self.loaded,
            "model": settings.model_path,
            "device": self.device,
            "dtype": settings.dtype,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "model_device_map": settings.model_device_map,
            "model_gpu_memory_utilization": settings.model_gpu_memory_utilization,
            "model_max_memory_gb": settings.model_max_memory_gb,
            "hf_device_map": hf_device_map,
            "sam2_device": settings.sam2_device,
            "clear_cuda_cache_after_run": settings.clear_cuda_cache_after_run,
        }


# Global inference engine instance
_inference_engine: Optional[Think2SegInference] = None
_inference_engine_lock = None


def _get_engine_lock():
    global _inference_engine_lock
    if _inference_engine_lock is None:
        import threading

        _inference_engine_lock = threading.Lock()
    return _inference_engine_lock


def get_inference_status() -> dict:
    """Return model/GPU status without forcing model initialization."""
    if _inference_engine is not None:
        return _inference_engine.get_status()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(gpu_count)
    ]
    return {
        "loaded": False,
        "model": settings.model_path,
        "device": settings.device,
        "dtype": settings.dtype,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "model_device_map": settings.model_device_map,
        "model_gpu_memory_utilization": settings.model_gpu_memory_utilization,
        "model_max_memory_gb": settings.model_max_memory_gb,
        "hf_device_map": None,
        "sam2_device": settings.sam2_device,
        "clear_cuda_cache_after_run": settings.clear_cuda_cache_after_run,
    }


def get_inference_engine(
    progress_callback: Optional[ProgressCallback] = None,
) -> Think2SegInference:
    """Get or create global inference engine."""
    global _inference_engine
    if _inference_engine is not None:
        return _inference_engine

    with _get_engine_lock():
        if _inference_engine is None:
            _inference_engine = Think2SegInference(
                progress_callback=progress_callback,
            )
    return _inference_engine
