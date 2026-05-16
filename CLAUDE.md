# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Macau-wide AI traffic signal control system. Uses a PPO reinforcement learning agent to control traffic lights across all intersections in Macau via SUMO simulation, with a parallel real-time computer vision pipeline that estimates congestion from live CCTV camera feeds (YOLO detection + tracking). Results are visualized on a Leaflet.js web dashboard.

## Key Technology Stack

- **Traffic simulator:** SUMO 1.26.0 (`sumo` and `sumo-gui` binaries), controlled via TraCI
- **RL framework:** Stable-Baselines3 PPO with Gymnasium environments
- **Object detection:** Ultralytics YOLO (yolo12l.pt, yolo26x.pt, etc.)
- **Web server:** FastAPI + Uvicorn (port 8000), frontend is a single Leaflet.js HTML page
- **Road network analysis:** OSMnx (graph-based heat diffusion for congestion propagation)
- **GPU:** RTX 3070 Ti (CUDA) when available, CPU fallback otherwise
- **Monitoring:** TensorBoard logs in `logs_v*/`

## Project Structure (by purpose)

```
log/training/         → Original SUMO Gym env + PPO training script (v3, city-scale 512x512x512 network)
log/                  → SUMO config (macao.sumocfg), route files (.rou.xml), traffic generation scripts, CSV logs
training_v4/          → Refined env with lane/phase caching for speed; GPU training script
test/                 → Model evaluation (score_model.py), multi-model comparison (compare.py, chart_heatmap_compare.py), traffic generators, SUMO env copy for testing
main/map_estimate.py  → Real-time 60-camera pipeline: YOLO detection → IoU tracking → congestion scoring → cloud push + CSV logging + optional UI mosaic grid
traffic-web/          → FastAPI server (server.py), Leaflet.js frontend (index.html), requirements.txt
1/ & 2/               → Earlier iterations of estimate.py and map_estimate.py (different YOLO models, different configs)
stream/               → Standalone YOLO stream viewer with line-crossing counters
SUMO/                 → SUMO installer, OSM→net.xml conversion, junction-to-GPS-coordinate mapper
models/               → Saved PPO checkpoints (v3, zipped)
v4_result/            → v4 GPU-trained checkpoints
v5_result/            → v5 lightweight CPU-trained checkpoints
result/               → v2 checkpoints (64x64 and 256x256 variants)
```

## Core Architecture

### RL Training Loop (SUMO + PPO)

1. **Environment** (`sumo_env.py` variants): A Gymnasium `Env` wrapping SUMO via TraCI. Action space is `MultiDiscrete([2] * num_traffic_lights)` — each intersection can be toggled to the next phase or kept. Observations are lane halting counts + current phase indices. Reward penalizes total halting vehicles and waiting time.
2. **Training** (`train.py` / `training.py`): Instantiates the env, configures a PPO model with MLP policy, trains with `CheckpointCallback` saving every N steps.
3. **Key environment constraint:** Only intersections with ≥2 phases are controllable. Yellow-light phases are auto-skipped. Minimum green time enforced before phase change allowed.

### Real-Time Vision Pipeline (`main/map_estimate.py`)

1. Reads up to 60 HLS/m3u8 CCTV stream URLs from `streams.txt`
2. Each stream gets a background thread continuously pulling frames via OpenCV
3. Main loop processes each camera at `PROC_FPS_PER_CAM` (5 FPS), running YOLO inference
4. For wide-angle cameras ( designated in `CAM_OVERRIDES`), motion detection via MOG2 background subtraction supplements YOLO detections
5. A simple IoU-based tracker maintains vehicle tracks across frames
6. Per-camera congestion score computed from: occupancy ratio, speed percentiles, stopped-vehicle ratio — with per-camera capacity scaling
7. Every 5 seconds: pushes GeoJSON to cloud API, writes CSV row
8. Optional mosaic UI shows all cameras in a grid with congestion bars

### Web Dashboard (`traffic-web/`)

- FastAPI serves `index.html` and two endpoints: `GET /api/traffic` (frontend polls every 5s) and `POST /api/update` (Python scripts push data)
- Frontend renders Macau road network as colored GeoJSON lines + camera markers, with HLS video popups on marker click

### Model Evaluation (`test/score_model.py`, `test/compare.py`)

- Loads a saved PPO model and runs it against a SUMO simulation for N steps
- Baseline mode (model_path=None) uses SUMO's default fixed-time signals
- Scoring formula: `(throughput * 20) - (avg_queue * 2) + (total_reward * 0.01)`
- Observation/action padding handles dimension mismatches between trained model and test environment

## Common Commands

```powershell
# Launch SUMO GUI with the Macau config
sumo-gui -c D:\school\bsd\log\macao.sumocfg

# Run PPO training (v4/v5 style)
python D:\school\bsd\training_v4\training.py

# Score a trained model against baseline
python D:\school\bsd\test\score_model.py

# Run multi-model comparison with charts
python D:\school\bsd\test\chart_heatmap_compare.py

# Start the traffic web dashboard
cd D:\school\bsd\traffic-web
pip install -r requirements.txt
python server.py
# Then open http://localhost:8000

# Run real-time 60-camera traffic estimation
python D:\school\bsd\main\map_estimate.py

# View TensorBoard training logs
tensorboard --logdir=D:\school\bsd\logs_v5

# Generate randomized SUMO route files for testing
python D:\school\bsd\test\traffic_generater.py
```

## Environment-Specific Notes

- **Requires `SUMO_HOME`** environment variable pointing to the SUMO installation directory
- SUMO binary must be accessible on PATH (`sumo`, `sumo-gui`)
- The SUMO config at `D:\school\bsd\log\macao.sumocfg` references network and route files via absolute paths — edit if relocating
- Camera stream URLs are stored in `streams.txt` files (one URL per line, `#` for comments)
- The Render-deployed cloud API URL is hardcoded in `main/map_estimate.py` as `CLOUD_API_URL`
- YOLO model files (`.pt`) are large and gitignored; they must be downloaded separately
- All Python scripts use hardcoded absolute paths starting with `D:\school\bsd\` — portable use requires path edits

## Key Files When Making Changes

| Change | Files to touch |
|--------|---------------|
| Modify RL reward/observation/action | `training_v4/sumo_env.py` (and matching copy in `test/sumo_env.py`) |
| Change PPO hyperparameters | `training_v4/training.py` or `log/training/train.py` |
| Add/change traffic light logic | `sumo_env.py` step() method in the relevant version |
| Change congestion scoring formula | `main/map_estimate.py` lines ~468-491 |
| Add camera overrides (wide-angle, etc.) | `CAM_OVERRIDES` dict in `main/map_estimate.py` |
| Change evaluation scoring formula | `test/score_model.py` base_score calculation |
| Modify web dashboard | `traffic-web/index.html` (frontend), `traffic-web/server.py` (backend) |
