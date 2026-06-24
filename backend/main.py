"""
Think2Seg FastAPI Backend
REST API for satellite image segmentation using natural language prompts.
"""

import base64
import io
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from config import settings
from handoff import CLASS_PROMPTS, SemanticMask, build_handoff_bundle
from inference import (
    DEFAULT_PROMPT_PROVIDER_CONFIG,
    DEFAULT_SEGMENTATION_OPTIONS,
    get_inference_engine,
    get_inference_status,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEGMENTATION_JOBS: dict[str, dict[str, Any]] = {}
SEGMENTATION_JOBS_LOCK = threading.Lock()
AETHER_JOBS: dict[str, dict[str, Any]] = {}
AETHER_JOBS_LOCK = threading.Lock()
SEGMENTATION_RUN_LOCK = threading.Lock()
MAX_JOB_HISTORY = 100
MAX_RESOURCE_LOG_ENTRIES = 240
MAX_RESOURCE_LOG_RESPONSE_ENTRIES = 120
TERMINAL_STATUSES = {"succeeded", "failed"}
_LAST_CPU_SAMPLE: tuple[int, int] | None = None


class SegmentationRequest(BaseModel):
    """Segmentation request model."""
    prompt: str


class SegmentationResponse(BaseModel):
    """Segmentation response model."""
    success: bool
    prompt: str
    mask: list = None
    visualization: str = None
    error: str = None


class RuntimeSettingsRequest(BaseModel):
    model_gpu_memory_utilization: float | None = None
    model_max_memory_gb: float | None = None


class PromptProviderSettingsRequest(BaseModel):
    prompt_provider: str | None = None
    litellm_model: str | None = None
    litellm_api_key: str | None = None
    litellm_api_base: str | None = None


class PostprocessRequest(BaseModel):
    options: dict[str, Any] | None = None
    selected_target_numbers: list[int] | None = None
    layer_number: int | None = None


class RefineOutputRequest(BaseModel):
    prompts: list[str] | None = None
    options: dict[str, Any] | None = None
    prompt_provider: dict[str, Any] | None = None


def _now() -> float:
    return time.time()


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    resource_log = job.get("resource_log", [])
    return {
        "id": job["id"],
        "filename": job.get("filename"),
        "prompt": job.get("prompt"),
        "prompts": job.get("prompts"),
        "prompt_provider": job.get("prompt_provider"),
        "options": job.get("options"),
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "result": job.get("result"),
        "resource_snapshot": job.get("resource_snapshot"),
        "resource_log": resource_log[-MAX_RESOURCE_LOG_RESPONSE_ENTRIES:],
        "resource_log_total": len(resource_log),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "updated_at": job.get("updated_at"),
    }


def _serialize_aether_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "filename": job.get("filename"),
        "crop_id": job.get("crop_id"),
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "result": job.get("result"),
        "handoff": job.get("handoff"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "updated_at": job.get("updated_at"),
    }


def _get_job(job_id: str) -> dict[str, Any] | None:
    with SEGMENTATION_JOBS_LOCK:
        job = SEGMENTATION_JOBS.get(job_id)
        return dict(job) if job else None


def _update_job(job_id: str, **updates: Any) -> None:
    with SEGMENTATION_JOBS_LOCK:
        job = SEGMENTATION_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _now()


def _get_aether_job(job_id: str) -> dict[str, Any] | None:
    with AETHER_JOBS_LOCK:
        job = AETHER_JOBS.get(job_id)
        return dict(job) if job else None


def _update_aether_job(job_id: str, **updates: Any) -> None:
    with AETHER_JOBS_LOCK:
        job = AETHER_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _now()


def _read_meminfo_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if not parts:
                    continue
                values[key] = int(parts[0]) * 1024
    except Exception:
        logger.debug("Could not read /proc/meminfo", exc_info=True)
    return values


def _read_process_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        logger.debug("Could not read process RSS", exc_info=True)
    return None


def _read_cpu_total_idle() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as stat_file:
            parts = stat_file.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        values = [int(value) for value in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except Exception:
        logger.debug("Could not read /proc/stat", exc_info=True)
    return None


def _cpu_usage_percent() -> float | None:
    global _LAST_CPU_SAMPLE
    current = _read_cpu_total_idle()
    if current is None:
        return None
    previous = _LAST_CPU_SAMPLE
    _LAST_CPU_SAMPLE = current
    if previous is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (1 - (idle_delta / total_delta)) * 100))


def _nvidia_smi_utilization() -> dict[int, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    values = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            values[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return values


def _collect_resource_snapshot() -> dict[str, Any]:
    meminfo = _read_meminfo_bytes()
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    snapshot: dict[str, Any] = {
        "time": _now(),
        "ram": {
            "total": total,
            "available": available,
            "used": used,
            "process_rss": _read_process_rss_bytes(),
        },
        "cpu": {
            "usage_percent": _cpu_usage_percent(),
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "count": os.cpu_count(),
        },
        "cuda_available": False,
        "gpus": [],
    }

    try:
        import torch

        snapshot["cuda_available"] = torch.cuda.is_available()
        gpu_utilization = _nvidia_smi_utilization()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    snapshot["gpus"].append({
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "free": free_bytes,
                        "total": total_bytes,
                        "used": total_bytes - free_bytes,
                        "allocated": torch.cuda.memory_allocated(index),
                        "reserved": torch.cuda.memory_reserved(index),
                        "utilization_percent": gpu_utilization.get(index),
                    })
    except Exception:
        logger.debug("Could not collect CUDA memory snapshot", exc_info=True)

    return snapshot


def _append_resource_log(job_id: str, stage: str, message: str) -> None:
    entry = {
        "time": _now(),
        "stage": stage,
        "message": message,
        "snapshot": _collect_resource_snapshot(),
    }
    with SEGMENTATION_JOBS_LOCK:
        job = SEGMENTATION_JOBS.get(job_id)
        if not job:
            return
        resource_log = job.setdefault("resource_log", [])
        resource_log.append(entry)
        if len(resource_log) > MAX_RESOURCE_LOG_ENTRIES:
            del resource_log[:-MAX_RESOURCE_LOG_ENTRIES]
        job["resource_snapshot"] = entry["snapshot"]
        job["updated_at"] = _now()


def _start_resource_monitor(job_id: str) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def monitor() -> None:
        while not stop_event.wait(2):
            job = _get_job(job_id)
            if not job or job.get("status") in TERMINAL_STATUSES:
                return
            _append_resource_log(
                job_id,
                job.get("stage") or "Running",
                job.get("message") or "Resource snapshot",
            )

    thread = threading.Thread(
        target=monitor,
        name=f"segment-resource-monitor-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _prune_jobs() -> None:
    with SEGMENTATION_JOBS_LOCK:
        if len(SEGMENTATION_JOBS) <= MAX_JOB_HISTORY:
            return

        finished = [
            job
            for job in SEGMENTATION_JOBS.values()
            if job.get("status") in TERMINAL_STATUSES
        ]
        finished.sort(key=lambda item: item.get("completed_at") or item["created_at"])
        remove_count = len(SEGMENTATION_JOBS) - MAX_JOB_HISTORY
        for job in finished[:remove_count]:
            SEGMENTATION_JOBS.pop(job["id"], None)


def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        return min(max(float(value), minimum), maximum)
    except (TypeError, ValueError):
        return fallback


def _clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return min(max(int(value), minimum), maximum)
    except (TypeError, ValueError):
        return fallback


def _build_segmentation_options(
    sam_mask_threshold: float,
    sam_multimask_output: bool,
    mask_min_area: int,
    mask_cleanup_px: int,
    mask_expand_px: int,
    refinement_passes: int,
    refinement_mode: str,
) -> dict[str, Any]:
    options = dict(DEFAULT_SEGMENTATION_OPTIONS)
    options.update({
        "sam_mask_threshold": _clamp_float(
            sam_mask_threshold,
            -2.0,
            2.0,
            DEFAULT_SEGMENTATION_OPTIONS["sam_mask_threshold"],
        ),
        "sam_multimask_output": bool(sam_multimask_output),
        "mask_min_area": _clamp_int(
            mask_min_area,
            0,
            2_000_000,
            DEFAULT_SEGMENTATION_OPTIONS["mask_min_area"],
        ),
        "mask_cleanup_px": _clamp_int(
            mask_cleanup_px,
            0,
            31,
            DEFAULT_SEGMENTATION_OPTIONS["mask_cleanup_px"],
        ),
        "mask_expand_px": _clamp_int(
            mask_expand_px,
            -31,
            31,
            DEFAULT_SEGMENTATION_OPTIONS["mask_expand_px"],
        ),
        "refinement_passes": _clamp_int(
            refinement_passes,
            0,
            3,
            DEFAULT_SEGMENTATION_OPTIONS["refinement_passes"],
        ),
        "refinement_mode": str(refinement_mode or "").lower(),
    })
    if options["refinement_mode"] not in {"intersection", "union", "replace"}:
        options["refinement_mode"] = DEFAULT_SEGMENTATION_OPTIONS["refinement_mode"]
    return options


def _normalize_prompt_provider_name(value: Any) -> str:
    provider = str(value or "local").lower().strip()
    if provider in {"byok", "litellm"}:
        provider = "cloud"
    if provider not in {"local", "cloud"}:
        provider = "local"
    return provider


def _build_prompt_provider_config(
    prompt_provider: str | None = None,
    litellm_model: str | None = None,
    litellm_api_key: str | None = None,
    litellm_api_base: str | None = None,
) -> dict[str, Any]:
    config = {
        "prompt_provider": _normalize_prompt_provider_name(
            prompt_provider if prompt_provider is not None else settings.prompt_provider,
        ),
        "litellm_model": str(
            litellm_model
            if litellm_model is not None
            else settings.litellm_model or ""
        ).strip(),
        "litellm_api_key": str(
            litellm_api_key
            if litellm_api_key is not None
            else settings.litellm_api_key or ""
        ).strip(),
        "litellm_api_base": str(
            litellm_api_base
            if litellm_api_base is not None
            else settings.litellm_api_base or ""
        ).strip(),
    }
    if config["prompt_provider"] == "cloud" and not config["litellm_model"]:
        raise HTTPException(
            status_code=400,
            detail="LiteLLM model is required when cloud prompt provider is selected",
        )
    return config


def _sanitize_prompt_provider_config(config: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = dict(DEFAULT_PROMPT_PROVIDER_CONFIG)
    if config:
        sanitized.update({
            "prompt_provider": _normalize_prompt_provider_name(config.get("prompt_provider")),
            "litellm_model": config.get("litellm_model") or "",
            "litellm_api_base": config.get("litellm_api_base") or "",
            "has_litellm_api_key": bool(config.get("litellm_api_key")),
        })
    return sanitized


def _prompt_provider_from_body(body: dict[str, Any] | None) -> dict[str, Any]:
    body = body or {}
    return _build_prompt_provider_config(
        body.get("prompt_provider"),
        body.get("litellm_model"),
        body.get("litellm_api_key"),
        body.get("litellm_api_base"),
    )


def _parse_prompt_layers(prompt: str, prompts_json: str | None = None) -> list[str]:
    prompts = []
    if prompts_json:
        try:
            parsed = json.loads(prompts_json)
            if isinstance(parsed, list):
                prompts.extend(str(item).strip() for item in parsed if str(item).strip())
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid prompts_json: {exc}") from exc

    if not prompts:
        prompts.extend(line.strip() for line in prompt.splitlines() if line.strip())

    if not prompts and prompt.strip():
        prompts = [prompt.strip()]

    deduped = []
    seen = set()
    for item in prompts:
        key = item.lower()
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def _encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _mask_to_png_base64(mask: np.ndarray) -> str:
    mask_uint8 = ((mask > 0) * 255).astype(np.uint8)
    return _encode_png_base64(Image.fromarray(mask_uint8))


def _output_root() -> Path:
    root = Path(settings.output_dir).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_output_dir(output_id: str) -> Path:
    root = _output_root().resolve()
    output_dir = (root / output_id).resolve()
    if root not in output_dir.parents and output_dir != root:
        raise HTTPException(status_code=400, detail="Invalid output id")
    return output_dir


def _output_file_url(output_id: str, filename: str) -> str:
    return f"/api/outputs/{quote(output_id, safe='')}/files/{quote(filename, safe='')}"


def _save_image_file(output_dir: Path, filename: str, image: Image.Image) -> str:
    image.save(output_dir / filename, format="PNG")
    return filename


def _save_mask_file(output_dir: Path, filename: str, mask: np.ndarray) -> str:
    mask_uint8 = ((mask > 0) * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(output_dir / filename, format="PNG")
    return filename


def _save_raw_mask_file(output_dir: Path, filename: str, mask: np.ndarray) -> str:
    np.save(output_dir / filename, np.asarray(mask))
    return filename


def _file_name_from_url(url: str | None) -> str | None:
    if not url or "/files/" not in url:
        return None
    return url.split("/files/", 1)[1]


def _load_target_mask(output_dir: Path, target: dict[str, Any], options: dict[str, Any]) -> np.ndarray | None:
    raw_file = target.get("raw_mask_file")
    if raw_file:
        raw_path = output_dir / raw_file
        if raw_path.exists():
            raw_mask = np.load(raw_path)
            if raw_mask.dtype == bool:
                return raw_mask.astype(np.uint8)
            return (raw_mask > options["sam_mask_threshold"]).astype(np.uint8)

    mask_file = _file_name_from_url(target.get("mask_url"))
    if mask_file:
        mask_path = output_dir / mask_file
        if mask_path.exists():
            return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
    return None


def _postprocess_mask_array(mask: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    processed = (mask > 0).astype(np.uint8)
    cleanup_px = int(options["mask_cleanup_px"])
    if cleanup_px > 0:
        kernel_size = cleanup_px if cleanup_px % 2 == 1 else cleanup_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    expand_px = int(options["mask_expand_px"])
    if expand_px:
        kernel_size = abs(expand_px) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if expand_px > 0:
            processed = cv2.dilate(processed, kernel, iterations=1)
        else:
            processed = cv2.erode(processed, kernel, iterations=1)

    min_area = int(options["mask_min_area"])
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


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _create_visualization_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]
    binary_mask = _resize_mask(mask, width, height)
    mask_colored = np.zeros_like(image_array)
    mask_colored[binary_mask > 0] = [42, 214, 255]
    blended = (0.62 * image_array.astype(float) + 0.38 * mask_colored.astype(float)).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blended, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
    return Image.fromarray(blended)


def _create_segmented_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image_array = np.array(image.convert("RGBA"))
    height, width = image_array.shape[:2]
    binary_mask = _resize_mask(mask, width, height)
    image_array[..., 3] = (binary_mask * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def _strip_base64(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_base64(item)
            for key, item in value.items()
            if not key.endswith("_base64")
        }
    if isinstance(value, list):
        return [_strip_base64(item) for item in value]
    return value


def _save_result_assets(
    output_id: str,
    output_dir: Path,
    prefix: str,
    results: dict[str, Any],
    response_data: dict[str, Any],
) -> None:
    if "mask" in results:
        filename = _save_mask_file(output_dir, f"{prefix}_mask.png", results["mask"])
        response_data["mask_url"] = _output_file_url(output_id, filename)
    if "visualization" in results:
        filename = _save_image_file(output_dir, f"{prefix}_overlay.png", results["visualization"])
        response_data["visualization_url"] = _output_file_url(output_id, filename)
    if "segmented_image" in results:
        filename = _save_image_file(
            output_dir,
            f"{prefix}_segmented.png",
            results["segmented_image"],
        )
        response_data["segmented_image_url"] = _output_file_url(output_id, filename)

    for index, (pass_result, pass_response) in enumerate(
        zip(results.get("pass_outputs", []), response_data.get("pass_outputs", [])),
        start=1,
    ):
        pass_prefix = f"{prefix}_pass_{index:02d}"
        if "mask" in pass_result:
            filename = _save_mask_file(output_dir, f"{pass_prefix}_mask.png", pass_result["mask"])
            pass_response["mask_url"] = _output_file_url(output_id, filename)
        if "visualization" in pass_result:
            filename = _save_image_file(
                output_dir,
                f"{pass_prefix}_overlay.png",
                pass_result["visualization"],
            )
            pass_response["visualization_url"] = _output_file_url(output_id, filename)
        if "segmented_image" in pass_result:
            filename = _save_image_file(
                output_dir,
                f"{pass_prefix}_segmented.png",
                pass_result["segmented_image"],
            )
            pass_response["segmented_image_url"] = _output_file_url(output_id, filename)

    for index, (target_result, target_response) in enumerate(
        zip(results.get("target_outputs", []), response_data.get("target_outputs", [])),
        start=1,
    ):
        target_prefix = f"{prefix}_target_{index:03d}"
        if "mask" in target_result:
            filename = _save_mask_file(output_dir, f"{target_prefix}_mask.png", target_result["mask"])
            target_response["mask_url"] = _output_file_url(output_id, filename)
        if "raw_mask" in target_result:
            filename = _save_raw_mask_file(
                output_dir,
                f"{target_prefix}_raw.npy",
                target_result["raw_mask"],
            )
            target_response["raw_mask_file"] = filename
        if "visualization" in target_result:
            filename = _save_image_file(
                output_dir,
                f"{target_prefix}_overlay.png",
                target_result["visualization"],
            )
            target_response["visualization_url"] = _output_file_url(output_id, filename)
        if "segmented_image" in target_result:
            filename = _save_image_file(
                output_dir,
                f"{target_prefix}_segmented.png",
                target_result["segmented_image"],
            )
            target_response["segmented_image_url"] = _output_file_url(output_id, filename)


def _save_segmentation_outputs(
    output_id: str,
    source_filename: str | None,
    prompt: str,
    source_image: Image.Image,
    results: dict[str, Any],
    response_data: dict[str, Any],
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    output_dir = _output_root() / output_id
    output_dir.mkdir(parents=True, exist_ok=True)

    response_data["output_id"] = output_id
    response_data["output_metadata_url"] = f"/api/outputs/{quote(output_id, safe='')}/metadata"

    original_name = _save_image_file(output_dir, "original.png", source_image.convert("RGB"))
    response_data["original_image_url"] = _output_file_url(output_id, original_name)

    _save_result_assets(output_id, output_dir, "final", results, response_data)
    for layer_result, layer_response in zip(
        results.get("layers", []),
        response_data.get("layers", []),
    ):
        layer_number = int(layer_result.get("layer_number") or layer_response.get("layer_number") or 0)
        _save_result_assets(
            output_id,
            output_dir,
            f"layer_{layer_number:02d}",
            layer_result,
            layer_response,
        )

    metadata = {
        "id": output_id,
        "filename": source_filename,
        "prompt": prompt,
        "prompts": results.get("prompts") or [prompt],
        "segmentation_generation": 1,
        "created_at": _now(),
        "result": _strip_base64(response_data),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _build_segment_response(results: dict[str, Any], prompt: str) -> dict[str, Any]:
    response_data = {
        "success": True,
        "prompt": prompt,
        "original_size": results["original_size"],
        "uploaded_size": results.get("uploaded_size"),
        "sam_prompts": results.get("sam_prompts"),
        "model_output": results.get("model_output"),
        "model_outputs": results.get("model_outputs"),
        "options": results.get("options"),
        "prompt_provider": results.get("prompt_provider"),
        "refinement_passes_completed": results.get("refinement_passes_completed"),
    }

    if "mask" in results:
        response_data["mask_base64"] = _mask_to_png_base64(results["mask"])

    if "visualization" in results:
        response_data["visualization_base64"] = _encode_png_base64(
            results["visualization"],
        )

    if "segmented_image" in results:
        response_data["segmented_image_base64"] = _encode_png_base64(
            results["segmented_image"],
        )

    pass_outputs = []
    for pass_output in results.get("pass_outputs", []):
        encoded_pass = {
            "pass_number": pass_output.get("pass_number"),
            "total_passes": pass_output.get("total_passes"),
            "prompt_count": pass_output.get("prompt_count"),
            "model_output": pass_output.get("model_output"),
            "sam_prompts": pass_output.get("sam_prompts"),
            "target_numbers": pass_output.get("target_numbers"),
        }
        if "mask" in pass_output:
            encoded_pass["mask_base64"] = _mask_to_png_base64(pass_output["mask"])
        if "visualization" in pass_output:
            encoded_pass["visualization_base64"] = _encode_png_base64(
                pass_output["visualization"],
            )
        if "segmented_image" in pass_output:
            encoded_pass["segmented_image_base64"] = _encode_png_base64(
                pass_output["segmented_image"],
            )
        pass_outputs.append(encoded_pass)

    if pass_outputs:
        response_data["pass_outputs"] = pass_outputs

    target_outputs = []
    for target_output in results.get("target_outputs", []):
        encoded_target = {
            "target_number": target_output.get("target_number"),
            "pass_number": target_output.get("pass_number"),
            "target_in_pass": target_output.get("target_in_pass"),
            "score": target_output.get("score"),
            "sam_prompt": target_output.get("sam_prompt"),
        }
        if "mask" in target_output:
            encoded_target["mask_base64"] = _mask_to_png_base64(target_output["mask"])
        if "visualization" in target_output:
            encoded_target["visualization_base64"] = _encode_png_base64(
                target_output["visualization"],
            )
        if "segmented_image" in target_output:
            encoded_target["segmented_image_base64"] = _encode_png_base64(
                target_output["segmented_image"],
            )
        target_outputs.append(encoded_target)

    if target_outputs:
        response_data["target_outputs"] = target_outputs

    layers = []
    for layer_result in results.get("layers", []):
        layer_response = _build_segment_response(
            layer_result,
            layer_result.get("prompt") or "",
        )
        layer_response["layer_number"] = layer_result.get("layer_number")
        layer_response["layer_prompt"] = layer_result.get("layer_prompt")
        layers.append(layer_response)

    if layers:
        response_data["layers"] = layers

    return response_data


def _build_layered_results(
    engine: Any,
    image: Image.Image,
    prompts: list[str],
    options: dict[str, Any],
    prompt_config: dict[str, Any] | None,
    progress_callback,
) -> dict[str, Any]:
    layer_results = []
    combined_mask = None
    all_targets = []
    all_prompts = []
    model_outputs = []
    processed_image = None

    for index, layer_prompt in enumerate(prompts):
        layer_start = int((index / len(prompts)) * 100)
        layer_end = int(((index + 1) / len(prompts)) * 100)

        def layer_progress(progress: int, message: str, *, start=layer_start, end=layer_end) -> None:
            scaled = start + int((max(0, min(100, progress)) / 100) * (end - start))
            progress_callback(scaled, f"Layer {index + 1}/{len(prompts)}: {message}")

        result = engine.segment(
            image,
            layer_prompt,
            return_visualization=True,
            progress_callback=layer_progress,
            options=options,
            prompt_config=prompt_config,
        )
        if not result.get("success"):
            raise RuntimeError(
                f"Layer '{layer_prompt}' failed: {result.get('error') or 'unknown error'}"
            )

        layer_number = index + 1
        target_offset = len(all_targets)
        result["layer_number"] = layer_number
        result["layer_prompt"] = layer_prompt
        for target in result.get("target_outputs", []):
            target["layer_number"] = layer_number
            target["layer_prompt"] = layer_prompt
            target["target_number"] = target_offset + int(target.get("target_number") or 0)
            all_targets.append(target)

        for pass_output in result.get("pass_outputs", []):
            pass_output["layer_number"] = layer_number
            pass_output["layer_prompt"] = layer_prompt
            pass_output["target_numbers"] = [
                target_offset + int(number)
                for number in pass_output.get("target_numbers", [])
            ]

        layer_results.append(result)
        all_prompts.extend(result.get("sam_prompts") or [])
        if result.get("model_output"):
            model_outputs.append(result["model_output"])
        processed_image = processed_image or engine.preprocess_image(image)
        layer_mask = np.asarray(result["mask"]).astype(np.uint8)
        combined_mask = (
            layer_mask
            if combined_mask is None
            else np.logical_or(combined_mask > 0, layer_mask > 0).astype(np.uint8)
        )

    if combined_mask is None:
        combined_mask = np.zeros(processed_image.size[::-1], dtype=np.uint8)

    return {
        "success": True,
        "prompt": " | ".join(prompts),
        "prompts": prompts,
        "mask": combined_mask,
        "original_size": processed_image.size,
        "uploaded_size": image.size,
        "sam_prompts": all_prompts,
        "model_output": "\n\n".join(model_outputs),
        "model_outputs": model_outputs,
        "options": options,
        "prompt_provider": _sanitize_prompt_provider_config(prompt_config),
        "refinement_passes_completed": max(
            (layer.get("refinement_passes_completed") or 1 for layer in layer_results),
            default=1,
        ),
        "target_outputs": all_targets,
        "layers": layer_results,
        "visualization": engine._create_visualization(processed_image, combined_mask),
        "segmented_image": engine._create_segmented_image(processed_image, combined_mask),
    }


def _run_segment_job(
    job_id: str,
    image_data: bytes,
    prompt: str,
    options: dict[str, Any],
    filename: str | None = None,
    prompts: list[str] | None = None,
    metadata_extra: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> None:
    _update_job(
        job_id,
        status="queued",
        progress=1,
        stage="Queued",
        message="Waiting for the segmentation worker",
    )
    _append_resource_log(job_id, "Queued", "Waiting for the segmentation worker")

    with SEGMENTATION_RUN_LOCK:
        stop_monitor, monitor_thread = _start_resource_monitor(job_id)
        try:
            def progress_callback(progress: int, message: str) -> None:
                job = _get_job(job_id) or {}
                next_progress = max(
                    int(job.get("progress") or 0),
                    max(0, min(99, int(progress))),
                )
                _update_job(
                    job_id,
                    status="running",
                    progress=next_progress,
                    stage=message,
                    message=message,
                )
                _append_resource_log(job_id, message, message)

            def scaled_progress(start: int, end: int) -> Any:
                span = end - start

                def _callback(progress: int, message: str) -> None:
                    scaled = start + int((max(0, min(100, progress)) / 100) * span)
                    progress_callback(scaled, message)

                return _callback

            _update_job(
                job_id,
                status="running",
                progress=4,
                stage="Validating image",
                message="Reading uploaded image",
                started_at=_now(),
            )
            _append_resource_log(job_id, "Validating image", "Reading uploaded image")
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if image.size[0] * image.size[1] > 25000000:
                raise ValueError("Image too large. Maximum ~5000x5000 pixels.")

            engine = get_inference_engine(
                progress_callback=scaled_progress(5, 45),
            )
            progress_callback(45, "Models ready")

            layer_prompts = prompts or [prompt]
            prompt_config = prompt_config or _build_prompt_provider_config()
            if len(layer_prompts) > 1:
                results = _build_layered_results(
                    engine,
                    image,
                    layer_prompts,
                    options,
                    prompt_config,
                    progress_callback=scaled_progress(45, 98),
                )
            else:
                results = engine.segment(
                    image,
                    layer_prompts[0],
                    return_visualization=True,
                    progress_callback=scaled_progress(45, 98),
                    options=options,
                    prompt_config=prompt_config,
                )
            if not results["success"]:
                raise RuntimeError(results.get("error") or "Segmentation failed")

            _update_job(
                job_id,
                status="running",
                progress=99,
                stage="Encoding result",
                message="Encoding mask and visualization",
            )
            _append_resource_log(job_id, "Encoding result", "Encoding mask and visualization")
            response_data = _build_segment_response(results, prompt)
            _save_segmentation_outputs(
                job_id,
                filename,
                prompt,
                image,
                results,
                response_data,
                metadata_extra=metadata_extra,
            )
            _update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="Complete",
                message="Segmentation complete",
                result=response_data,
                prompt_provider=_sanitize_prompt_provider_config(prompt_config),
                completed_at=_now(),
            )
            logger.info("Segmentation job %s completed", job_id)
        except Exception as exc:
            logger.error("Segmentation job %s failed: %s", job_id, exc)
            _append_resource_log(job_id, "Failed", str(exc))
            _update_job(
                job_id,
                status="failed",
                progress=100,
                stage="Failed",
                message=str(exc),
                error=str(exc),
                completed_at=_now(),
            )
        finally:
            stop_monitor.set()
            monitor_thread.join()
            _append_resource_log(job_id, "Finished", "Final resource snapshot")


def _callback_url(crop_id: str) -> str | None:
    base_url = (settings.ultra_sim_callback_base_url or "").strip().rstrip("/")
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("ULTRA_SIM_CALLBACK_BASE_URL must be an absolute http(s) URL")
    return f"{base_url}/api/aether/handoff/{quote(crop_id, safe='')}"


def _push_handoff(crop_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
    callback_url = _callback_url(crop_id)
    if callback_url is None:
        return {
            "status": "not_configured",
            "message": "ULTRA_SIM_CALLBACK_BASE_URL is not configured",
        }

    headers = {"Content-Type": "application/json"}
    if settings.ultra_sim_callback_token:
        headers["X-Aether-Handoff-Token"] = settings.ultra_sim_callback_token
    request = Request(
        callback_url,
        data=json.dumps(bundle, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
        return {
            "status": "accepted",
            "http_status": response.status,
            "local_job_id": response_payload.get("job_id"),
            "status_url": response_payload.get("status_url"),
        }


def _run_aether_job(
    job_id: str,
    image_data: bytes,
    crop_id: str,
    filename: str,
) -> None:
    _update_aether_job(
        job_id,
        status="queued",
        progress=1,
        stage="Queued",
        message="Waiting for the AETHER worker",
        started_at=_now(),
    )
    with SEGMENTATION_RUN_LOCK:
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if image.size[0] * image.size[1] > 25000000:
                raise ValueError("Image too large. Maximum ~5000x5000 pixels.")

            def load_progress(progress: int, message: str) -> None:
                _update_aether_job(
                    job_id,
                    status="running",
                    progress=max(2, min(5, int(progress * 0.05))),
                    stage=message,
                    message="Loading shared inference models",
                )

            engine = get_inference_engine(progress_callback=load_progress)
            semantic_masks: list[SemanticMask] = []
            class_items = list(CLASS_PROMPTS.items())
            class_count = len(class_items)

            for index, (class_name, class_prompt) in enumerate(class_items):
                start = 5 + int((index / class_count) * 82)
                end = 5 + int(((index + 1) / class_count) * 82)

                def progress_callback(
                    progress: int,
                    message: str,
                    *,
                    progress_start: int = start,
                    progress_end: int = end,
                    current_class: str = class_name,
                    class_index: int = index,
                ) -> None:
                    scaled = progress_start + int(
                        (max(0, min(100, progress)) / 100)
                        * (progress_end - progress_start)
                    )
                    _update_aether_job(
                        job_id,
                        status="running",
                        progress=min(scaled, 94),
                        stage=f"{current_class}: {message}",
                        message=f"Class {class_index + 1} of {class_count}",
                    )

                result = engine.segment(
                    image,
                    class_prompt,
                    return_visualization=False,
                    progress_callback=progress_callback,
                )
                if not result.get("success"):
                    raise RuntimeError(
                        f"{class_name} inference failed: "
                        f"{result.get('error') or 'unknown error'}"
                    )
                semantic_masks.append(
                    SemanticMask(
                        class_name=class_name,
                        prompt=class_prompt,
                        mask=np.asarray(result["mask"]).astype(np.uint8),
                    )
                )

            _update_aether_job(
                job_id,
                status="running",
                progress=95,
                stage="Building handoff",
                message="Deduplicating masks and calculating class vectors",
            )
            processed_image = engine.preprocess_image(image)
            bundle = build_handoff_bundle(
                crop_id=crop_id,
                source_filename=filename,
                source_image_bytes=image_data,
                image=processed_image,
                semantic_masks=semantic_masks,
                min_component_area=settings.aether_min_component_area,
                dedup_iou_threshold=settings.aether_dedup_iou_threshold,
                dedup_containment_threshold=settings.aether_dedup_containment_threshold,
                classification_smoothing=settings.aether_classification_smoothing,
            )
            _update_aether_job(job_id, bundle=bundle)
            _update_aether_job(
                job_id,
                status="running",
                progress=98,
                stage="Pushing to Ultra-Sim",
                message="Sending masks and classifications to the laptop",
            )
            handoff = _push_handoff(crop_id, bundle)
            result_summary = {
                "success": True,
                "crop_id": crop_id,
                "segment_count": bundle["segment_count"],
                "classification_counts": bundle["classification_counts"],
                "visualization_base64": bundle["whole_scene_overlay_base64"],
            }
            _update_aether_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="Complete",
                message=(
                    "Masks accepted by Ultra-Sim"
                    if handoff["status"] == "accepted"
                    else "Inference complete; laptop callback is not configured"
                ),
                result=result_summary,
                handoff=handoff,
                bundle=bundle,
                completed_at=_now(),
            )
        except Exception as exc:
            logger.exception("AETHER handoff job %s failed", job_id)
            _update_aether_job(
                job_id,
                status="failed",
                progress=100,
                stage="Failed",
                message=str(exc),
                error=str(exc),
                completed_at=_now(),
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    logger.info("Think2Seg API starting...")
    logger.info("Model will be loaded lazily on the first segmentation job.")
    yield
    logger.info("Think2Seg API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Think2Seg Demo API",
    description="Natural Language Satellite Image Segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = get_inference_status()
    return {
        "status": "healthy",
        "model_loaded": status["loaded"],
        "model": settings.model_path,
    }


@app.get("/status")
async def get_status():
    """Get model status."""
    return get_inference_status()


@app.get("/runtime-settings")
async def get_runtime_settings():
    """Return resource allocation settings used for the next model load."""
    status = get_inference_status()
    return {
        "model_loaded": status["loaded"],
        "model_gpu_memory_utilization": settings.model_gpu_memory_utilization,
        "model_max_memory_gb": settings.model_max_memory_gb,
        "applies_to_loaded_model": not status["loaded"],
    }


@app.get("/prompt-provider-settings")
async def get_prompt_provider_settings():
    """Return default prompt-generation provider settings without exposing keys."""
    return {
        "prompt_provider": _normalize_prompt_provider_name(settings.prompt_provider),
        "litellm_model": settings.litellm_model or "",
        "litellm_api_base": settings.litellm_api_base or "",
        "has_litellm_api_key": bool(settings.litellm_api_key),
    }


@app.post("/prompt-provider-settings")
async def update_prompt_provider_settings(request: PromptProviderSettingsRequest):
    """Update prompt-generation provider defaults for future jobs."""
    if request.prompt_provider is not None:
        settings.prompt_provider = _normalize_prompt_provider_name(request.prompt_provider)
    if request.litellm_model is not None:
        settings.litellm_model = request.litellm_model.strip() or None
    if request.litellm_api_base is not None:
        settings.litellm_api_base = request.litellm_api_base.strip() or None
    if request.litellm_api_key is not None:
        settings.litellm_api_key = request.litellm_api_key.strip() or None

    return {
        "prompt_provider": _normalize_prompt_provider_name(settings.prompt_provider),
        "litellm_model": settings.litellm_model or "",
        "litellm_api_base": settings.litellm_api_base or "",
        "has_litellm_api_key": bool(settings.litellm_api_key),
        "message": "Prompt provider settings updated",
    }


@app.post("/runtime-settings")
async def update_runtime_settings(request: RuntimeSettingsRequest):
    """Update resource allocation settings for future model loads."""
    if request.model_gpu_memory_utilization is not None:
        settings.model_gpu_memory_utilization = min(
            max(float(request.model_gpu_memory_utilization), 0.1),
            0.99,
        )
    if request.model_max_memory_gb is not None:
        value = float(request.model_max_memory_gb)
        settings.model_max_memory_gb = value if value > 0 else None

    status = get_inference_status()
    return {
        "model_loaded": status["loaded"],
        "model_gpu_memory_utilization": settings.model_gpu_memory_utilization,
        "model_max_memory_gb": settings.model_max_memory_gb,
        "message": (
            "Settings will apply on the next model load"
            if status["loaded"]
            else "Settings will apply to the first model load"
        ),
    }


@app.get("/outputs")
async def list_outputs():
    """List saved segmentation output folders."""
    outputs = []
    for metadata_path in _output_root().glob("*/metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            result = metadata.get("result") or {}
            outputs.append({
                "id": metadata.get("id") or metadata_path.parent.name,
                "filename": metadata.get("filename"),
                "prompt": metadata.get("prompt"),
                "prompts": metadata.get("prompts"),
                "segmentation_generation": metadata.get("segmentation_generation", 1),
                "root_output_id": metadata.get("root_output_id"),
                "parent_output_id": metadata.get("parent_output_id"),
                "created_at": metadata.get("created_at"),
                "original_image_url": result.get("original_image_url"),
                "visualization_url": result.get("visualization_url"),
                "segmented_image_url": result.get("segmented_image_url"),
                "target_count": len(result.get("target_outputs") or []),
                "pass_count": len(result.get("pass_outputs") or []),
                "result": result,
            })
        except Exception:
            logger.warning("Could not read output metadata %s", metadata_path, exc_info=True)
    outputs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"outputs": outputs}


@app.get("/outputs/{output_id}/metadata")
async def get_output_metadata(output_id: str):
    """Return metadata for one saved segmentation output."""
    metadata_path = _safe_output_dir(output_id) / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Saved output not found")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@app.get("/outputs/{output_id}/files/{filename:path}")
async def get_output_file(output_id: str, filename: str):
    """Serve a saved output image from a segmentation run."""
    output_dir = _safe_output_dir(output_id).resolve()
    file_path = (output_dir / filename).resolve()
    if output_dir not in file_path.parents and file_path != output_dir:
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(file_path)


@app.post("/outputs/{output_id}/postprocess")
async def postprocess_output(output_id: str, request: PostprocessRequest):
    """Recompose saved target masks with new post-processing settings."""
    output_dir = _safe_output_dir(output_id)
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Saved output not found")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    result = metadata.get("result") or {}
    options = dict(DEFAULT_SEGMENTATION_OPTIONS)
    options.update(result.get("options") or {})
    options.update(request.options or {})
    options = _build_segmentation_options(
        options["sam_mask_threshold"],
        options["sam_multimask_output"],
        options["mask_min_area"],
        options["mask_cleanup_px"],
        options["mask_expand_px"],
        options["refinement_passes"],
        options["refinement_mode"],
    )

    targets = result.get("target_outputs") or []
    if request.layer_number is not None:
        targets = [
            target
            for target in targets
            if int(target.get("layer_number") or 0) == int(request.layer_number)
        ]
    if request.selected_target_numbers is not None:
        selected = {int(number) for number in request.selected_target_numbers}
        targets = [
            target
            for target in targets
            if int(target.get("target_number") or 0) in selected
        ]

    combined_mask = None
    target_responses = []
    original = Image.open(output_dir / "original.png").convert("RGB")
    for target in targets:
        mask = _load_target_mask(output_dir, target, options)
        if mask is None:
            continue
        mask = _postprocess_mask_array(mask, options)
        combined_mask = mask if combined_mask is None else np.logical_or(combined_mask > 0, mask > 0).astype(np.uint8)
        target_response = dict(target)
        target_response["mask_base64"] = _mask_to_png_base64(mask)
        target_response["visualization_base64"] = _encode_png_base64(
            _create_visualization_image(original, mask),
        )
        target_response["segmented_image_base64"] = _encode_png_base64(
            _create_segmented_image(original, mask),
        )
        target_responses.append(target_response)

    if combined_mask is None:
        combined_mask = np.zeros((original.height, original.width), dtype=np.uint8)
    combined_mask = _postprocess_mask_array(combined_mask, options)
    return {
        "success": True,
        "output_id": output_id,
        "options": options,
        "selected_target_numbers": request.selected_target_numbers,
        "mask_base64": _mask_to_png_base64(combined_mask),
        "visualization_base64": _encode_png_base64(
            _create_visualization_image(original, combined_mask),
        ),
        "segmented_image_base64": _encode_png_base64(
            _create_segmented_image(original, combined_mask),
        ),
        "target_outputs": target_responses,
    }


@app.post("/outputs/{output_id}/refine/jobs")
async def refine_output_job(
    output_id: str,
    request: RefineOutputRequest,
    background_tasks: BackgroundTasks,
):
    """Create a segmentation job from a saved segmented output."""
    output_dir = _safe_output_dir(output_id)
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Saved output not found")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    result = metadata.get("result") or {}
    image_name = _file_name_from_url(result.get("segmented_image_url")) or "final_segmented.png"
    image_path = output_dir / image_name
    if not image_path.exists():
        image_path = output_dir / "original.png"
    prompts = [prompt.strip() for prompt in (request.prompts or metadata.get("prompts") or [metadata.get("prompt")]) if str(prompt).strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts available for refinement")
    base_options = dict(DEFAULT_SEGMENTATION_OPTIONS)
    base_options.update(result.get("options") or {})
    base_options.update(request.options or {})
    options = _build_segmentation_options(
        base_options["sam_mask_threshold"],
        base_options["sam_multimask_output"],
        base_options["mask_min_area"],
        base_options["mask_cleanup_px"],
        base_options["mask_expand_px"],
        0,
        base_options["refinement_mode"],
    )
    prompt_config = _prompt_provider_from_body(request.prompt_provider)
    job_id = uuid.uuid4().hex
    image_data = image_path.read_bytes()
    clean_prompt = " | ".join(prompts)
    generation = int(metadata.get("segmentation_generation") or 1) + 1
    job = {
        "id": job_id,
        "filename": f"refine-{metadata.get('filename') or output_id}.png",
        "prompt": clean_prompt,
        "prompts": prompts,
        "prompt_provider": _sanitize_prompt_provider_config(prompt_config),
        "options": options,
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "message": "Waiting to refine saved output",
        "error": None,
        "result": None,
        "resource_snapshot": None,
        "resource_log": [],
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "updated_at": _now(),
    }
    with SEGMENTATION_JOBS_LOCK:
        SEGMENTATION_JOBS[job_id] = job
    _prune_jobs()
    metadata_extra = {
        "parent_output_id": output_id,
        "root_output_id": metadata.get("root_output_id") or output_id,
        "segmentation_generation": generation,
    }
    background_tasks.add_task(
        _run_segment_job,
        job_id,
        image_data,
        clean_prompt,
        options,
        job["filename"],
        prompts,
        metadata_extra,
        prompt_config,
    )
    return JSONResponse(_serialize_job(job))


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    prompts_json: str | None = Form(None),
    prompt_provider: str | None = Form(None),
    litellm_model: str | None = Form(None),
    litellm_api_key: str | None = Form(None),
    litellm_api_base: str | None = Form(None),
    sam_mask_threshold: float = Form(DEFAULT_SEGMENTATION_OPTIONS["sam_mask_threshold"]),
    sam_multimask_output: bool = Form(DEFAULT_SEGMENTATION_OPTIONS["sam_multimask_output"]),
    mask_min_area: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_min_area"]),
    mask_cleanup_px: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_cleanup_px"]),
    mask_expand_px: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_expand_px"]),
    refinement_passes: int = Form(DEFAULT_SEGMENTATION_OPTIONS["refinement_passes"]),
    refinement_mode: str = Form(DEFAULT_SEGMENTATION_OPTIONS["refinement_mode"]),
):
    """
    Segment satellite image based on natural language prompt.

    Args:
        file: Uploaded image file
        prompt: Natural language description of what to segment

    Returns:
        Segmentation results with mask and visualization
    """
    try:
        # Validate file
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and validate image
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        if image.size[0] * image.size[1] > 25000000:  # ~5000x5000
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum ~5000x5000 pixels.",
            )

        # Get inference engine
        engine = get_inference_engine()
        prompts = _parse_prompt_layers(prompt, prompts_json)
        if not prompts:
            raise HTTPException(status_code=400, detail="Prompt is required")
        options = _build_segmentation_options(
            sam_mask_threshold,
            sam_multimask_output,
            mask_min_area,
            mask_cleanup_px,
            mask_expand_px,
            refinement_passes,
            refinement_mode,
        )
        prompt_config = _build_prompt_provider_config(
            prompt_provider,
            litellm_model,
            litellm_api_key,
            litellm_api_base,
        )

        # Run segmentation
        if len(prompts) > 1:
            results = _build_layered_results(
                engine,
                image,
                prompts,
                options,
                prompt_config,
                progress_callback=lambda _progress, _message: None,
            )
        else:
            results = engine.segment(
                image,
                prompts[0],
                return_visualization=True,
                options=options,
                prompt_config=prompt_config,
            )

        if not results["success"]:
            raise HTTPException(status_code=500, detail=results.get("error"))

        display_prompt = " | ".join(prompts)
        response_data = _build_segment_response(results, display_prompt)
        _save_segmentation_outputs(
            uuid.uuid4().hex,
            file.filename,
            display_prompt,
            image,
            results,
            response_data,
        )
        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in segment endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/jobs")
async def create_segment_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    prompts_json: str | None = Form(None),
    prompt_provider: str | None = Form(None),
    litellm_model: str | None = Form(None),
    litellm_api_key: str | None = Form(None),
    litellm_api_base: str | None = Form(None),
    sam_mask_threshold: float = Form(DEFAULT_SEGMENTATION_OPTIONS["sam_mask_threshold"]),
    sam_multimask_output: bool = Form(DEFAULT_SEGMENTATION_OPTIONS["sam_multimask_output"]),
    mask_min_area: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_min_area"]),
    mask_cleanup_px: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_cleanup_px"]),
    mask_expand_px: int = Form(DEFAULT_SEGMENTATION_OPTIONS["mask_expand_px"]),
    refinement_passes: int = Form(DEFAULT_SEGMENTATION_OPTIONS["refinement_passes"]),
    refinement_mode: str = Form(DEFAULT_SEGMENTATION_OPTIONS["refinement_mode"]),
):
    """Create an asynchronous segmentation job and return its status URL."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    prompts = _parse_prompt_layers(prompt, prompts_json)
    if not prompts:
        raise HTTPException(status_code=400, detail="Prompt is required")
    clean_prompt = " | ".join(prompts)

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="File is empty")

    options = _build_segmentation_options(
        sam_mask_threshold,
        sam_multimask_output,
        mask_min_area,
        mask_cleanup_px,
        mask_expand_px,
        refinement_passes,
        refinement_mode,
    )
    prompt_config = _build_prompt_provider_config(
        prompt_provider,
        litellm_model,
        litellm_api_key,
        litellm_api_base,
    )
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "filename": file.filename,
        "prompt": clean_prompt,
        "prompts": prompts,
        "prompt_provider": _sanitize_prompt_provider_config(prompt_config),
        "options": options,
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "message": "Waiting to start",
        "error": None,
        "result": None,
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "updated_at": _now(),
    }
    with SEGMENTATION_JOBS_LOCK:
        SEGMENTATION_JOBS[job_id] = job
    _prune_jobs()

    background_tasks.add_task(
        _run_segment_job,
        job_id,
        image_data,
        clean_prompt,
        options,
        file.filename,
        prompts,
        None,
        prompt_config,
    )
    return JSONResponse(_serialize_job(job))


@app.get("/segment/jobs/{job_id}")
async def get_segment_job(job_id: str):
    """Get progress and result for an asynchronous segmentation job."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Segmentation job not found")
    return JSONResponse(_serialize_job(job))


@app.get("/segment/jobs")
async def list_segment_jobs():
    """List recent segmentation jobs."""
    with SEGMENTATION_JOBS_LOCK:
        jobs = [_serialize_job(job) for job in SEGMENTATION_JOBS.values()]
    jobs.sort(key=lambda item: item["created_at"], reverse=True)
    return {"jobs": jobs}


@app.post("/aether/jobs")
async def create_aether_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    crop_id: str = Form(...),
):
    """Run all broad-class prompts and push masks/classification to Ultra-Sim."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    clean_crop_id = crop_id.strip()
    if not clean_crop_id or len(clean_crop_id) > 64:
        raise HTTPException(status_code=400, detail="A valid crop_id is required")
    if not all(character.isalnum() or character in "_-" for character in clean_crop_id):
        raise HTTPException(status_code=400, detail="crop_id contains unsupported characters")
    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="File is empty")

    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "filename": file.filename or "combined.png",
        "crop_id": clean_crop_id,
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "message": "Waiting to start",
        "error": None,
        "result": None,
        "handoff": None,
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "updated_at": _now(),
    }
    with AETHER_JOBS_LOCK:
        AETHER_JOBS[job_id] = job
    background_tasks.add_task(
        _run_aether_job,
        job_id,
        image_data,
        clean_crop_id,
        file.filename or "combined.png",
    )
    return JSONResponse(_serialize_aether_job(job))


@app.get("/aether/jobs/{job_id}")
async def get_aether_job(job_id: str):
    """Return one remote AETHER handoff job."""
    job = _get_aether_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="AETHER job not found")
    return JSONResponse(_serialize_aether_job(job))


@app.get("/aether/jobs")
async def list_aether_jobs():
    """List remote AETHER handoff jobs without returning mask bundles."""
    with AETHER_JOBS_LOCK:
        jobs = [_serialize_aether_job(job) for job in AETHER_JOBS.values()]
    jobs.sort(key=lambda item: item["created_at"], reverse=True)
    return {"jobs": jobs}


@app.post("/aether/jobs/{job_id}/handoff")
async def retry_aether_handoff(job_id: str):
    """Retry delivery of a completed bundle to Ultra-Sim."""
    job = _get_aether_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="AETHER job not found")
    bundle = job.get("bundle")
    if not isinstance(bundle, dict):
        raise HTTPException(status_code=409, detail="AETHER bundle is not ready")
    try:
        handoff = _push_handoff(str(job["crop_id"]), bundle)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    _update_aether_job(job_id, handoff=handoff)
    return handoff


@app.post("/segment-multiple")
async def segment_multiple(
    file: UploadFile = File(...),
    prompts: str = Form(...),  # Comma-separated prompts
):
    """
    Segment satellite image with multiple natural language prompts.

    Args:
        file: Uploaded image file
        prompts: Comma-separated natural language descriptions

    Returns:
        List of segmentation results
    """
    try:
        # Parse prompts
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No valid prompts provided")

        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")

        engine = get_inference_engine()

        # Process each prompt
        results = []
        for prompt in prompt_list:
            result = engine.segment(image, prompt, return_visualization=False)
            results.append(result)

        return JSONResponse({"success": True, "results": results})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in segment-multiple endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get API configuration."""
    return {
        "model_name": settings.model_name,
        "model_path": settings.model_path,
        "device": settings.device,
        "dtype": settings.dtype,
        "max_image_size": settings.max_image_size,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info" if settings.debug else "warning",
    )
