# robot-sim

AI-assisted robotics simulator. Solo, zero-funding build.

Core principle: the AI layer explains and orchestrates; a real physics
engine (PyBullet) computes. The LLM never invents a physics number.

## Current scope (v1)
Robot type: serial robotic arms only. Input format: structured parameters, not CAD upload.
Pipeline: ArmConfig -> URDF generator -> PyBullet simulation -> (soon) LLM report.

## Setup
1. pip install -r requirements.txt   (pybullet compiles from source - takes several minutes, this is normal)
2. pytest tests/ -v   (should show 3 passed)
3. uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
4. Visit /health and /demo/two-link-arm
