"""
API skeleton. For now: a health check, and one endpoint that runs the full
config -> URDF -> simulation pipeline so you can hit it from the browser
preview in Codespace and see it's actually working.
"""

from fastapi import FastAPI, HTTPException

from src.urdf_generator.generator import generate_urdf
from src.urdf_generator.samples import two_link_arm
from src.simulation.runner import run_smoke_test

app = FastAPI(title="Robot Sim API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo/two-link-arm")
def demo_two_link_arm():
    """
    Full pipeline smoke test: builds the sample 2-link arm config,
    generates URDF, runs it in pybullet, returns real simulation output.
    """
    try:
        config = two_link_arm()
        urdf = generate_urdf(config)
        result = run_smoke_test(urdf)
        return {
            "config_name": config.name,
            "urdf_preview": urdf[:300] + "...",
            "simulation_result": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
