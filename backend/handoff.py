"""AETHER mask/classification bundle generation for Ultra-Sim handoff."""

from __future__ import annotations

import base64
import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCHEMA_VERSION = "aether_remote_handoff_v1"
OPEN_SET_LABEL = "unknown_segment"
SOURCE_MODEL = "nicholas-edu/isr-segmentation"

CLASS_PROMPTS: dict[str, str] = {
    "aircraft": "aircraft, airplanes, and helicopters",
    "aviation_infrastructure": "airport runways, taxiways, aprons, hangars, and helipads",
    "bridge": "bridges and elevated crossings",
    "built_structure": "buildings and built structures",
    "facility": "industrial, utility, military, and institutional facilities",
    "harbor_infrastructure": "ports, docks, piers, quays, and harbor infrastructure",
    "roundabout": "road roundabouts and traffic circles",
    "ship": "ships, boats, and vessels",
    "sports_area": "stadiums, courts, tracks, fields, and sports areas",
    "storage_tank": "storage tanks, fuel tanks, and silos",
    "vehicle": "cars, trucks, buses, and ground vehicles",
    "vertical_infrastructure": "towers, masts, cranes, chimneys, and vertical infrastructure",
}
CLASS_NAMES = tuple(CLASS_PROMPTS)


@dataclass(frozen=True)
class SemanticMask:
    class_name: str
    prompt: str
    mask: np.ndarray


def _encode_mask(mask: np.ndarray) -> str:
    image = Image.fromarray(((mask > 0) * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _component_masks(
    semantic_masks: list[SemanticMask],
    min_area: int,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for semantic in semantic_masks:
        binary = (semantic.mask > 0).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        for label_index in range(1, count):
            if int(stats[label_index, cv2.CC_STAT_AREA]) >= min_area:
                candidates.append((labels == label_index).astype(np.uint8))
    return sorted(candidates, key=lambda mask: int(mask.sum()), reverse=True)


def _mask_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(left > 0, right > 0).sum())
    if intersection <= 0:
        return 0.0, 0.0
    left_area = max(int(left.sum()), 1)
    right_area = max(int(right.sum()), 1)
    union = left_area + right_area - intersection
    return intersection / max(union, 1), intersection / min(left_area, right_area)


def _deduplicate_masks(
    candidates: list[np.ndarray],
    iou_threshold: float,
    containment_threshold: float,
) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for candidate in candidates:
        match_index = None
        for index, existing in enumerate(merged):
            iou, containment = _mask_overlap(candidate, existing)
            if iou >= iou_threshold or containment >= containment_threshold:
                match_index = index
                break
        if match_index is None:
            merged.append(candidate)
        else:
            merged[match_index] = np.logical_or(
                merged[match_index] > 0,
                candidate > 0,
            ).astype(np.uint8)
    return merged


def _class_scores(
    mask: np.ndarray,
    semantic_masks: list[SemanticMask],
) -> dict[str, float]:
    area = max(int(mask.sum()), 1)
    scores = dict.fromkeys(CLASS_NAMES, 0.0)
    for semantic in semantic_masks:
        overlap = int(np.logical_and(mask > 0, semantic.mask > 0).sum())
        scores[semantic.class_name] = max(
            scores[semantic.class_name],
            overlap / area,
        )
    return scores


def _normalize_scores(
    scores: dict[str, float],
    smoothing: float,
) -> dict[str, float]:
    adjusted = {
        class_name: max(float(scores.get(class_name, 0.0)), 0.0) + smoothing
        for class_name in CLASS_NAMES
    }
    total = sum(adjusted.values())
    if total <= 0:
        return {class_name: 1.0 / len(CLASS_NAMES) for class_name in CLASS_NAMES}
    return {
        class_name: adjusted[class_name] / total
        for class_name in CLASS_NAMES
    }


def _geometry(mask: np.ndarray) -> dict[str, Any] | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < 25:
        return None
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    centroid_x = moments["m10"] / moments["m00"] if moments["m00"] else x + width / 2
    centroid_y = moments["m01"] / moments["m00"] if moments["m00"] else y + height / 2
    perimeter = float(cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * perimeter), True)
    return {
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_width": int(width),
        "bbox_height": int(height),
        "area_px": int(mask.sum()),
        "contour_area_px": round(contour_area, 3),
        "centroid_x": round(float(centroid_x), 3),
        "centroid_y": round(float(centroid_y), 3),
        "perimeter_px": round(perimeter, 3),
        "polygon": [
            [int(point[0][0]), int(point[0][1])]
            for point in approx
        ],
    }


def _render_overlay(image: Image.Image, masks: list[np.ndarray]) -> str:
    image_array = np.asarray(image.convert("RGB")).copy()
    overlay = image_array.copy()
    palette = (
        (51, 214, 198),
        (91, 164, 255),
        (246, 182, 74),
        (167, 139, 250),
    )
    for index, mask in enumerate(masks):
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        active = mask > 0
        overlay[active] = (
            0.52 * overlay[active].astype(np.float32)
            + 0.48 * color.astype(np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_handoff_bundle(
    *,
    crop_id: str,
    source_filename: str,
    source_image_bytes: bytes,
    image: Image.Image,
    semantic_masks: list[SemanticMask],
    min_component_area: int = 50,
    dedup_iou_threshold: float = 0.55,
    dedup_containment_threshold: float = 0.85,
    classification_smoothing: float = 0.01,
) -> dict[str, Any]:
    """Create object masks and full broad-class vectors for Ultra-Sim."""
    candidates = _component_masks(semantic_masks, min_component_area)
    masks = _deduplicate_masks(
        candidates,
        dedup_iou_threshold,
        dedup_containment_threshold,
    )
    image_area = max(image.width * image.height, 1)
    segments: list[dict[str, Any]] = []
    rendered_masks: list[np.ndarray] = []
    class_counts = dict.fromkeys(CLASS_NAMES, 0)

    for index, mask in enumerate(masks):
        geometry = _geometry(mask)
        if geometry is None:
            continue
        scores = _class_scores(mask, semantic_masks)
        probabilities = _normalize_scores(scores, classification_smoothing)
        ranking = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_class, top_probability = ranking[0]
        second_probability = ranking[1][1] if len(ranking) > 1 else 0.0
        class_counts[top_class] += 1
        rendered_masks.append(mask)
        segments.append(
            {
                "segment_id": f"{crop_id}_seg_{index:04d}",
                "label": OPEN_SET_LABEL,
                "mask_base64": _encode_mask(mask),
                "geometry": geometry,
                "area_ratio": round(geometry["area_px"] / image_area, 8),
                "class_scores": {
                    key: round(value, 8)
                    for key, value in scores.items()
                },
                "probabilities": {
                    key: round(value, 8)
                    for key, value in probabilities.items()
                },
                "top_class": top_class,
                "top_probability": round(top_probability, 8),
                "confidence_margin": round(
                    top_probability - second_probability,
                    8,
                ),
                "matched_prompts": [
                    semantic.prompt
                    for semantic in semantic_masks
                    if scores.get(semantic.class_name, 0.0) > 0
                ],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "crop_id": crop_id,
        "source": {
            "model": SOURCE_MODEL,
            "filename": source_filename,
            "sha256": hashlib.sha256(source_image_bytes).hexdigest(),
        },
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "classes": list(CLASS_NAMES),
        "segment_count": len(segments),
        "classification_counts": class_counts,
        "whole_scene_overlay_base64": _render_overlay(image, rendered_masks),
        "segments": segments,
    }
