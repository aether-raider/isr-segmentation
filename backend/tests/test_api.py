import io
import sys
import os
import base64
from types import SimpleNamespace
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

    def segment(
        self,
        image: Image.Image,
        prompt: str,
        return_visualization: bool = True,
        progress_callback=None,
    ):
        if progress_callback:
            progress_callback(25, "Generating spatial prompts")
            progress_callback(90, "Rendering visualization")
        # Return a simple black mask and a red visualization of same size
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[8:32, 8:32] = 1
        viz = Image.new("RGB", (w, h), color=(255, 0, 0))
        return {"success": True, "prompt": prompt, "mask": mask, "original_size": image.size, "visualization": viz}

    def preprocess_image(self, image: Image.Image):
        return image.convert("RGB")


@pytest.fixture(autouse=True)
def patch_inference(monkeypatch):
    # Patch the inference module to use a fake engine to avoid heavy model loads
    import inference as inf

    fake = FakeEngine()
    # Set the module-level engine and factory
    inf._inference_engine = fake
    monkeypatch.setattr(
        inf,
        "get_inference_engine",
        lambda *args, **kwargs: inf._inference_engine,
    )
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


def test_segment_job_endpoint_returns_progress_and_result():
    import main

    client = TestClient(main.app)

    img = Image.new("RGB", (64, 48), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/segment/jobs",
        files={"file": ("test.png", buf, "image/png")},
        data={"prompt": "buildings"},
    )
    assert res.status_code == 200
    job_id = res.json()["id"]

    status_res = client.get(f"/segment/jobs/{job_id}")
    assert status_res.status_code == 200
    job = status_res.json()
    assert job["status"] == "succeeded"
    assert job["progress"] == 100
    assert job["result"]["success"] is True
    assert "visualization_base64" in job["result"]


def test_aether_job_builds_classification_bundle_without_configured_callback(monkeypatch):
    import main

    monkeypatch.setattr(main.settings, "ultra_sim_callback_base_url", None)
    main.AETHER_JOBS.clear()
    client = TestClient(main.app)
    img = Image.new("RGB", (64, 48), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/aether/jobs",
        files={"file": ("test.png", buf, "image/png")},
        data={"crop_id": "crop-alpha"},
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    job = client.get(f"/aether/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["segment_count"] == 1
    assert sum(job["result"]["classification_counts"].values()) == 1
    assert job["handoff"]["status"] == "not_configured"
    assert "bundle" not in job


def test_segment_generates_sam_prompt_and_predicts_mask():
    import inference as inf

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(pad_token_id=0)
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return FakeBatch({"input_ids": inf.torch.tensor([[1, 2]])})

        def batch_decode(self, completion_ids, **kwargs):
            return [
                '<think>found the target</think><answer>```json'
                '[{"bbox_2d": [1, 1, 8, 8], '
                '"positive_points": [[3, 3], [4, 4]]}]'
                "```</answer>"
            ]

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return inf.torch.tensor([[1, 2, 3, 4]])

    class FakePredictor:
        def __init__(self):
            self.image_shape = None
            self.predict_kwargs = None

        def set_image(self, image):
            self.image_shape = image.shape[:2]

        def predict(self, **kwargs):
            self.predict_kwargs = kwargs
            mask = np.zeros(self.image_shape, dtype=bool)
            mask[2:8, 2:8] = True
            return np.array([mask]), None, None

    engine = inf.Think2SegInference.__new__(inf.Think2SegInference)
    engine.loaded = True
    engine.input_device = inf.torch.device("cpu")
    engine.processor = FakeProcessor()
    engine.model = FakeModel()
    engine.sam2_predictor = FakePredictor()
    engine.sam2_error = None

    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    result = engine.segment(image, "buildings", return_visualization=False)

    assert result["success"] is True
    assert result["mask"].sum() == 36
    assert result["sam_prompts"][0]["bbox_2d"] == [1, 1, 8, 8]
    assert "input_ids" in engine.model.kwargs
    assert engine.sam2_predictor.predict_kwargs["box"].tolist() == [1, 1, 8, 8]
