"""
API skeleton. For now: a health check, and one endpoint that runs the full
config -> URDF -> simulation pipeline so you can hit it from the browser
preview in Codespace and see it's actually working.
"""

from fastapi import FastAPI, HTTPException

from src.urdf_generator.generator import generate_urdf
from src.urdf_generator.samples import two_link_arm
from src.simulation.runner import run_smoke_test
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test
from src.ai_layer.report_generator import generate_report

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


@app.get("/demo/two-link-arm/torque-check")
def demo_two_link_arm_torque_check():
    """
    Pure-math static capability check for the sample arm - no simulation.
    Computes required torque at each joint to hold the arm fully extended
    horizontally (worst case for gravity loading) and compares it to each
    joint's max_torque_nm.
    """
    try:
        config = two_link_arm()
        results = compute_static_torques(config)
        return {"config_name": config.name, "torque_check": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo/two-link-arm/lift-test")
def demo_two_link_arm_lift_test():
    """
    Dynamic counterpart to the static torque check: actually commands the
    arm's motors to hold the fully-extended horizontal pose under gravity
    and payload, and reports what really happened (sag, actual torque draw).
    """
    try:
        config = two_link_arm()
        result = run_lift_test(config)
        return {"config_name": config.name, "lift_test": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo/two-link-arm/report")
def demo_two_link_arm_report():
    """
    Full pipeline: real static torque check + real dynamic lift test,
    both fed to the LLM as data, which writes a plain-English design
    review grounded strictly in those numbers. Requires ANTHROPIC_API_KEY
    to be set in the environment.
    """
    try:
        config = two_link_arm()
        torque_results = compute_static_torques(config)
        lift_results = run_lift_test(config)
        report_text = generate_report(config.name, torque_results, lift_results)
        return {
            "config_name": config.name,
            "torque_check": torque_results,
            "lift_test": lift_results,
            "report": report_text,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
