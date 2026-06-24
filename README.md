# ISR Segmentation - Natural Language Satellite Image Segmentation

A modern full-stack demo application showcasing capabilities for segmenting satellite imagery using natural language prompts.

## Features

- **Natural Language Interface**: Describe what you want to segment in plain English
- **Real-time Processing**: Fast satellite image segmentation powered by Think2Seg-RS
- **Modern UI**: Beautiful, responsive React frontend with Tailwind CSS
- **REST API**: FastAPI backend with comprehensive documentation
- **Docker Support**: One-command deployment with docker-compose
- **GPU Accelerated**: CUDA support for faster inference
- **Multi-Format Support**: Works with PNG, JPG, TIFF images

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Vue 3/React)               │
│  • Image Upload (Drag & Drop)                           │
│  • Natural Language Input                               │
│  • Results Visualization                                │
│  • Download Functionality                               │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI/Python)                   │
│  • Image Preprocessing                                  │
│  • Think2Seg-RS Inference                               │
│  • SAM2 Segmentation                                    │
│  • Result Post-processing                               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          ML MODELS (CUDA/GPU)                           │
│  • Qwen-2.5-VL (LVLM for reasoning)                     │
│  • SAM2 (Segmentation Model)                            │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### For Local Development
- **Python 3.10+**: Backend runtime
- **Node.js 18+**: Frontend build
- **CUDA 12.2+**: GPU support (optional but recommended)
- **Docker & Docker Compose**: For containerized deployment

### System Requirements
- **GPU**: NVIDIA GPU with ≥8GB VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB system RAM minimum
- **Storage**: 50GB for models + workspace

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Navigate to the demo folder
cd <project-directory>

# 2. Build and start both services
docker-compose up --build

# 3. Open the app in your browser
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Docker Compose commands

```bash
# Rebuild the backend and frontend images
docker-compose build --no-cache

# Start services in detached mode
docker-compose up -d

# View backend logs
docker-compose logs -f backend

# Stop and remove containers
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v
```

### Notes

- If you are using an NVIDIA GPU, make sure Docker is configured for GPU support.
- The backend currently defaults to `DEVICE=cuda` in `backend/.env`.
- For CPU-only setups, change `DEVICE=cpu` and use a compatible CPU PyTorch install.

### Option 2: Local Development

#### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env

# Update .env with your configuration:
# - MODEL_PATH: HuggingFace model ID or local path
# - DEVICE: cuda or cpu
# - SAM2_MODEL_SIZE: tiny, small, base_plus, large

# Run server
python main.py
```

Server will be available at `http://localhost:8000`

#### Frontend Setup

```bash
# Navigate to frontend (in new terminal)
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:5173`

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - Model status
- `GET /config` - API configuration

### Segmentation
- `POST /segment` - Single image segmentation
  ```bash
  curl -X POST "http://localhost:8000/segment" \
    -F "file=@image.jpg" \
    -F "prompt=buildings in urban areas"
  ```

- `POST /aether/jobs` - Run all 12 AETHER broad classes and push the artifact
  bundle to an existing Ultra-Sim crop on the configured laptop callback.
- `GET /aether/jobs/{job_id}` - Poll remote segmentation, classification, and
  callback progress.
- `POST /aether/jobs/{job_id}/handoff` - Retry delivery of a completed bundle.

- `POST /segment-multiple` - Multiple prompts on single image
  ```bash
  curl -X POST "http://localhost:8000/segment-multiple" \
    -F "file=@image.jpg" \
    -F "prompts=buildings,roads,vegetation"
  ```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

**Backend** (`.env`):
```ini
# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model
MODEL_NAME=Think2Seg-RS-7B
MODEL_PATH=OpenGVLab/Think2Seg-RS-7B
DEVICE=cuda
DTYPE=bfloat16

# SAM2
SAM2_MODEL_SIZE=base_plus
SAM2_DEVICE=cuda

# Performance
MAX_IMAGE_SIZE=1024
INFERENCE_TIMEOUT=300
USE_CACHE=true
```

**Frontend** (`.env`):
```ini
VITE_API_URL=http://localhost:8000
```

## Development

### Backend Development

```bash
cd backend

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black pylint

# Format code
black .

# Run linter
pylint *.py

# Run tests (if available)
pytest tests/
```

### Frontend Development

```bash
cd frontend

# Install dev dependencies
npm install

# Run with HMR
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

## Model Information

### Think2Seg-RS
- **Architecture**: Qwen-2.5-VL + SAM2
- **Training Dataset**: EarthReason
- **Parameters**: 3B or 7B
- **Inference Speed**: ~2-5 seconds per image (depends on size)
- **HuggingFace**: `OpenGVLab/Think2Seg-RS-7B`

### SAM2
- **Model Size Options**: tiny, small, base_plus, large
- **Recommended**: base_plus (good balance of speed/accuracy)
- **Facebook Research**: https://github.com/facebookresearch/sam2
 
## Model Weights & Download

By default the backend uses HuggingFace's `from_pretrained()` API (see `backend/config.py`). That means:

- Automatic download: If `MODEL_PATH` in `backend/.env` is a HuggingFace model ID (for example `OpenGVLab/Think2Seg-RS-7B`) the container or local process will download the weights automatically the first time the server starts (internet required).
- Cache location: downloaded files are cached under the model cache directory configured in `backend/.env` (default: `.model_cache`). When running with Docker Compose the demo maps a named volume to `/app/.model_cache` so downloads persist across container restarts.

When you may prefer to download weights manually (offline environments, faster startup, or to provide private tokens), follow one of the options below.

1) Let Docker download automatically (recommended)

 - Ensure the host has internet access.
 - If the model is private, set your HuggingFace token in the environment so the container can access it:

```bash
# on the host (PowerShell)
setx HUGGINGFACE_HUB_TOKEN "<your_token>"
# or add to backend/.env: HUGGINGFACE_HUB_TOKEN=<your_token>
```

 - Start the stack:

```bash
docker-compose up --build
```

The backend will download the model into the cache directory and load it on startup.

2) Pre-download the model on the host and mount it into the container

 - Install `huggingface_hub` on your host (or use an environment that has it):

```bash
pip install huggingface_hub
huggingface-cli login   # if model is private
```

 - Use `snapshot_download` to pull the model to a local folder (example):

```bash
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("OpenGVLab/Think2Seg-RS-7B", cache_dir="./model_cache/Think2Seg-RS-7B")
PY
```

 - Update `backend/.env` to point `MODEL_PATH` at the downloaded folder (inside the container). If you plan to mount `./model_cache` into the container at `/app/.model_cache`, set:

```ini
MODEL_PATH=/app/.model_cache/Think2Seg-RS-7B
```

 - Update `docker-compose.yml` (backend service) to mount the folder (example snippet):

```yaml
services:
  backend:
    volumes:
      - ./model_cache:/app/.model_cache
```

 - Then start the stack:

```bash
docker-compose up --build
```

3) Local development (no Docker)

 - If running locally (not in Docker), download the model into `backend/.model_cache/Think2Seg-RS-7B` and set `MODEL_PATH=backend/.model_cache/Think2Seg-RS-7B` in your local `.env` before running `python main.py`.

Notes
- Disk usage: Think2Seg model weights can be large (several GB). Ensure you have ~30–50GB free for models and caches.
- Private models: use `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN` so the process/container can authenticate.
- If downloads fail in Docker, check the container logs (`docker-compose logs backend`) to see the exact error.

## UI Features

### Image Upload
- Drag & drop support
- Click to browse
- Size validation
- Format support: PNG, JPG, TIFF

### Natural Language Input
- Free-form text prompts
- Suggested prompts for common use cases
- Context-aware help text

### Results Display
- Original image
- Segmentation mask overlay
- Downloadable results
- Processing statistics

### Example Prompts
- "buildings in urban areas"
- "forest coverage and vegetation"
- "roads and highway networks"
- "water bodies and lakes"
- "agricultural fields"
- "industrial structures"

## Docker Deployment

### Build Images

```bash
# Build both images
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Run Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Remove volumes
docker-compose down -v
```

### GPU Support

Ensure Docker has GPU support:

```bash
# Ubuntu/Linux
sudo apt-get install nvidia-docker2
nvidia-docker --version

# Run with GPU
docker run --gpus all ...
```

## Performance Optimization

### For Faster Inference
1. Use smaller SAM2 model: `tiny` or `small`
2. Reduce image size: Set `MAX_IMAGE_SIZE=512`
3. Enable vLLM: Set `USE_VLLM=true` (requires extra setup)
4. Use bfloat16: Faster on newer GPUs (RTX 40-series, H100)

### Memory Management
- **GPU VRAM**: ~12GB for 7B model
- **System RAM**: ~16GB minimum
- **Model Cache**: ~30GB on disk

## Troubleshooting

### Model Loading Issues
```
Issue: "Could not download model"
Solution: 
- Check HuggingFace token: huggingface-cli login
- Or specify local path: MODEL_PATH=/path/to/model
```

### CUDA/GPU Issues
```
Issue: "CUDA out of memory"
Solution:
- Reduce image size in config
- Use smaller model (3B instead of 7B)
- Use smaller SAM2 variant (tiny)
```

### Port Already in Use
```
# Change API port
export API_PORT=8001

# Change frontend port (in vite.config.js)
server: { port: 5174 }
```

## Project Structure

```
Think2Seg-Demo/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── inference.py            # Think2Seg wrapper
│   ├── config.py               # Configuration
│   ├── requirements.txt         # Dependencies
│   └── .env.example            # Example config
├── frontend/
│   ├── src/
│   │   ├── App.vue             # Main component
│   │   ├── main.js             # Entry point
│   │   └── style.css           # Styles
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Build config
│   ├── tailwind.config.js       # Tailwind config
│   └── nginx.conf              # Nginx config
├── docker-compose.yml          # Docker Compose
├── Dockerfile.backend          # Backend image
├── Dockerfile.frontend         # Frontend image
└── README.md                   # This file
```

## References

- **Think2Seg-RS Paper**: ISPRS Journal of Photogrammetry and Remote Sensing
- **GitHub**: [Open-R1 Repository](https://github.com/OpenGVLab/Open-R1)
- **Models**: [HuggingFace Hub](https://huggingface.co/OpenGVLab)

## Contributing

To improve this demo:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Important Notes

1. **First Run**: Model download may take 5-10 minutes on first startup
2. **VRAM Requirements**: 7B model requires ~12GB VRAM
3. **Internet Connection**: Required for model download from HuggingFace
4. **Production Deployment**: Consider using Kubernetes, add authentication, implement caching

---

Built using FastAPI, Vue.js, and Tailwind CSS
