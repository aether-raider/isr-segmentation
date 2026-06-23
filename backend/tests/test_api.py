import io
import sys
import os
import base64
import numpy as np
from PIL import Image
import pytest

# Ensure backend folder is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient


class FakeEngine:
    def __init__(self):
        self.loaded = True

    def get_status(self):
        return {"loaded": True, "model": "fake-model", "device": "cpu", "dtype": "float32"}

    def segment(self, image: Image.Image, prompt: str, return_visualization: bool = True):
        # Return a simple black mask and a red visualization of same size
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        viz = Image.new("RGB", (w, h), color=(255, 0, 0))
        return {"success": True, "prompt": prompt, "mask": mask, "original_size": image.size, "visualization": viz}


@pytest.fixture(autouse=True)
def patch_inference(monkeypatch):
    # Patch the inference module to use a fake engine to avoid heavy model loads
    import inference as inf

    fake = FakeEngine()
    # Set the module-level engine and factory
    inf._inference_engine = fake
    monkeypatch.setattr(inf, "get_inference_engine", lambda: inf._inference_engine)
    yield


def test_health_endpoint():
    import main

    client = TestClient(main.app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_segment_endpoint_returns_visualization():
    import main

    client = TestClient(main.app)

    # Create small test image
    img = Image.new("RGB", (64, 48), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    data = {"prompt": "buildings"}

    res = client.post("/segment", files=files, data=data)
    assert res.status_code == 200
    data = res.json()
    assert data.get("success") is True
    assert "visualization_base64" in data

    # Validate that the visualization can be decoded
    viz_b64 = data["visualization_base64"]
    decoded = base64.b64decode(viz_b64)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
