"""
API skeleton. For now: a health check, and one endpoint that runs the full
config -> URDF -> simulation pipeline so you can hit it from the browser
preview in Codespace and see it's actually working.
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.urdf_generator.schema import ArmConfig
from src.urdf_generator.generator import generate_urdf, validate_config
from src.urdf_generator.samples import two_link_arm
from src.simulation.runner import run_smoke_test
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test
from src.ai_layer.report_generator import generate_report
from src.storage import config_store
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from src.analysis.mesh_processor import compute_mesh_properties, MeshProcessingError
from src.analysis.step_processor import compute_step_properties, StepProcessingError
from src.storage.mesh_store import save_mesh_file
from src.components.motor_library import list_motors
from src.simulation.motion_planner import plan_reach_motion
from src.analysis.inverse_kinematics import IKError
from src.analysis.mass_optimizer import optimize_link_masses
from src.urdf_generator.assembly import build_config_from_meshes, AssemblyError
from src.ml.predict import predict_lift_test, PredictionError
from src.auth.auth import (
    create_user, authenticate_user, create_access_token, get_current_user_id,
)
from src.auth.rate_limit import limiter

app = FastAPI(title="Robot Sim API")
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429, content={"detail": "Rate limit exceeded - please slow down."}
    ),
)
app.add_middleware(SlowAPIMiddleware)

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50MB - generous for a single-part mesh/CAD file, bounds worst-case memory/disk use per upload


@app.on_event("startup")
def _init_database():
    from src.storage.database import init_db
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
@limiter.limit("5/minute")
def signup(request: Request, body: dict):
    """
    body: {"email": str, "password": str}
    Returns an access token immediately on success, same as /auth/login -
    so a new user doesn't need a separate login call right after signing up.
    """
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email is required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    try:
        user = create_user(email, password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    token = create_access_token(user["user_id"])
    return {"access_token": token, "token_type": "bearer", "email": email}


@app.post("/auth/login")
@limiter.limit("5/minute")
def login(request: Request, body: dict):
    """body: {"email": str, "password": str}"""
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    try:
        user_id = authenticate_user(email, password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    token = create_access_token(user_id)
    return {"access_token": token, "token_type": "bearer", "email": email}

@app.get("/demo/two-link-arm")
def demo_two_link_arm(user_id: str = Depends(get_current_user_id)):
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
def demo_two_link_arm_torque_check(user_id: str = Depends(get_current_user_id)):
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
def demo_two_link_arm_lift_test(user_id: str = Depends(get_current_user_id)):
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
def demo_two_link_arm_report(user_id: str = Depends(get_current_user_id)):
    """
    Full pipeline: real static torque check + real dynamic lift test,
    both fed to the LLM as data, which writes a plain-English design
    review grounded strictly in those numbers. Requires GEMINI_API_KEY
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


@app.post("/analyze/arm")
@limiter.limit("20/minute")
def analyze_arm(
    config: ArmConfig,
    include_lift_test: bool = Query(True, description="Run the dynamic PyBullet lift test (slower, ~3 sim-seconds)."),
    include_report: bool = Query(False, description="Generate an LLM report - requires GEMINI_API_KEY, and uses your free-tier quota."),
    include_trajectory: bool = Query(False, description="Record a joint-angle trajectory (for 3D animation playback). Only used if include_lift_test is also true."),
    user_id: str = Depends(get_current_user_id),
    request: Request = None,
):
    """
    The real entry point: submit ANY arm design (not just the built-in
    sample) as an ArmConfig JSON body, and get back the full analysis.

    Note on the shoulder axis: use [0, 1, 0] or [1, 0, 0] (perpendicular to
    the link) if you want that joint to actually resist gravity when lifting.
    An axis of [0, 0, 1] makes it a roll/twist joint instead - see the
    two_link_arm sample's shoulder joint for a real example of that
    distinction, found during dynamic testing.
    """
    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})

    try:
        torque_results = compute_static_torques(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = {
        "config_name": config.name,
        "torque_check": torque_results,
    }

    if include_lift_test:
        try:
            response["lift_test"] = run_lift_test(config, record_trajectory=include_trajectory)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Lift test failed: {exc}")

    if include_report:
        try:
            response["report"] = generate_report(
                config.name, torque_results, response.get("lift_test")
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    return response


@app.post("/designs")
def save_design_endpoint(config: ArmConfig, user_id: str = Depends(get_current_user_id)):
    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})
    return config_store.save_design(config, user_id)


@app.get("/designs")
def list_designs_endpoint(user_id: str = Depends(get_current_user_id)):
    return config_store.list_designs(user_id)


@app.get("/designs/{slug}")
def get_design_endpoint(slug: str, user_id: str = Depends(get_current_user_id)):
    try:
        config = config_store.load_design(slug, user_id)
        return config.model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/designs/{slug}")
def delete_design_endpoint(slug: str, user_id: str = Depends(get_current_user_id)):
    deleted = config_store.delete_design(slug, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No saved design found for '{slug}'")
    return {"deleted": True, "slug": slug}


@app.post("/designs/from_meshes")
@limiter.limit("20/minute")
def build_design_from_meshes_endpoint(request: Request, body: dict, user_id: str = Depends(get_current_user_id)):
    """
    Assembles a full ArmConfig from already-uploaded mesh_ids, pulling
    length/mass/COM/inertia straight from what /meshes/upload computed and
    stored - no manual re-typing of geometry into an ArmConfig by hand.

    Body shape:
    {
      "name": str,
      "link_specs": [{"name": str, "mesh_id": str}, ...],
      "joint_specs": [{"name": str, "joint_type": "revolute", "axis": [0,1,0],
                        "lower_limit_rad": float, "upper_limit_rad": float,
                        "max_torque_nm": float}, ...],
      "payload_mass_kg": float (optional, default 0.0)
    }
    joint_specs[i] connects link_specs[i-1] (or base_link, for i=0) to
    link_specs[i] - one joint per link, in chain order.
    """
    try:
        config = build_config_from_meshes(
            name=body["name"],
            link_specs=body["link_specs"],
            joint_specs=body["joint_specs"],
            user_id=user_id,
            payload_mass_kg=body.get("payload_mass_kg", 0.0),
        )
    except AssemblyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing required field: {exc}")

    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})

    return config.model_dump()

@app.post("/predict/lift_test")
@limiter.limit("30/minute")
def predict_lift_test_endpoint(request: Request, config: ArmConfig, user_id: str = Depends(get_current_user_id)):
    """
    Fast, ML-based approximation of the dynamic lift test - no real
    simulation is run. Only supports 2-link arms named upper_arm/forearm
    with joints named shoulder/elbow (the shape the models were trained
    on). Torque and pass/fail predictions are well-validated; sag
    magnitude for failing joints is a rough, low-confidence estimate -
    see predict.py's docstring. Not a substitute for /analyze/arm.
    """
    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})

    try:
        return predict_lift_test(config)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/optimize/mass")
def optimize_mass_endpoint(
    config: ArmConfig,
    safety_margin: float = Query(0.2, ge=0, lt=1, description="Required torque margin after optimization."),
    min_mass_kg: float = Query(0.1, gt=0, description="Floor on any individual link's mass."),
    user_id: str = Depends(get_current_user_id),
):
    """
    Finds the lightest possible link masses (via a real linear program) that
    keep every joint within the requested safety margin. Does not change
    link lengths, joint torque ratings, or payload - only link masses.
    """
    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})
    return optimize_link_masses(config, safety_margin=safety_margin, min_mass_kg=min_mass_kg)


@app.post("/meshes/upload")
@limiter.limit("10/minute")
async def upload_mesh(request: Request, file: UploadFile = File(...), target_mass_kg: float = Form(...), user_id: str = Depends(get_current_user_id)):
    """Uploads an STL mesh, computes real mass properties via trimesh, and
    stores the file. Returns everything needed to build a mesh-based Link."""
    mesh_bytes = await file.read()
    if len(mesh_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large - max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.")
    try:
        props = compute_mesh_properties(mesh_bytes, file_type="stl", target_mass_kg=target_mass_kg)
    except MeshProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response_body = {
        "length_m": props["effective_length_m"],
        "mass_kg": props["mass_kg"],
        "com_offset_m": props["com_offset_z_m"],
        "is_watertight": props["is_watertight"],
        "inertia_ixx": props["inertia_tensor"]["ixx"],
        "inertia_iyy": props["inertia_tensor"]["iyy"],
        "inertia_izz": props["inertia_tensor"]["izz"],
        "inertia_ixy": props["inertia_tensor"]["ixy"],
        "inertia_ixz": props["inertia_tensor"]["ixz"],
        "inertia_iyz": props["inertia_tensor"]["iyz"],
    }
    # Save the NORMALIZED mesh (origin shifted to the proximal face), not
    # the raw upload - otherwise the rendered/collision geometry would be
    # positioned inconsistently with the computed com_offset_m/inertia.
    # Persisted under the mesh_id so build_config_from_meshes() can look
    # these up later without the original upload response being replayed.
    mesh_id = save_mesh_file(props["normalized_mesh_bytes"], user_id, properties=response_body)
    return {"mesh_id": mesh_id, **response_body}

@app.post("/meshes/upload_step")
@limiter.limit("10/minute")
async def upload_step(request: Request, file: UploadFile = File(...), target_mass_kg: float = Form(...), user_id: str = Depends(get_current_user_id)):
    """
    Uploads a real STEP CAD file, converts it to a mesh-backed link via
    cadquery/OpenCASCADE (see step_processor.py), and stores it exactly
    like an STL upload - the returned mesh_id works identically in
    /designs/from_meshes, the Quick Assembly frontend, and everywhere
    else a mesh_id is used. STEP files with more than one solid are
    rejected (one part per file, same convention as STL uploads).
    """
    step_bytes = await file.read()
    if len(step_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large - max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.")
    try:
        props = compute_step_properties(step_bytes, target_mass_kg=target_mass_kg)
    except StepProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response_body = {
        "length_m": props["effective_length_m"],
        "mass_kg": props["mass_kg"],
        "com_offset_m": props["com_offset_z_m"],
        "is_watertight": props["is_watertight"],
        "inertia_ixx": props["inertia_tensor"]["ixx"],
        "inertia_iyy": props["inertia_tensor"]["iyy"],
        "inertia_izz": props["inertia_tensor"]["izz"],
        "inertia_ixy": props["inertia_tensor"]["ixy"],
        "inertia_ixz": props["inertia_tensor"]["ixz"],
        "inertia_iyz": props["inertia_tensor"]["iyz"],
        "source_format": "step",
    }
    mesh_id = save_mesh_file(props["normalized_mesh_bytes"], user_id, properties=response_body)
    return {"mesh_id": mesh_id, **response_body}


@app.get("/meshes/{mesh_id}/file")
def get_mesh_file(mesh_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Serves the raw STL bytes for a previously uploaded mesh - needed so the
    browser's 3D viewer (three.js STLLoader) can fetch and render the real
    geometry, not just receive its computed properties as JSON. Ownership
    is verified via load_mesh_properties() (raises FileNotFoundError, same
    as a nonexistent mesh, for another user's mesh - doesn't leak existence).
    """
    from src.storage.mesh_store import MESH_STORAGE_DIR, load_mesh_properties
    import re

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", mesh_id)

    try:
        load_mesh_properties(safe_id, user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No mesh found for '{mesh_id}'")

    path = MESH_STORAGE_DIR / f"{safe_id}.stl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No mesh found for '{mesh_id}'")

    return FileResponse(path, media_type="model/stl", filename=f"{safe_id}.stl")


@app.get("/components/motors")
def list_motors_endpoint(user_id: str = Depends(get_current_user_id)):
    """Returns the real motor library, for populating a picker in the UI."""
    return list_motors()



@app.post("/motion/reach")
def plan_reach_endpoint(
    config: ArmConfig,
    target_x: float = Query(..., description="Target x position in meters"),
    target_y: float = Query(..., description="Target y position in meters"),
    target_z: float = Query(..., description="Target z position in meters"),
    tracking_tolerance_rad: float = Query(0.05, gt=0, description="How close a joint must get to its target angle to count as successful"),
    record_trajectory: bool = Query(True, description="Record joint-angle trajectory for 3D animation playback"),
    user_id: str = Depends(get_current_user_id),
):
    """
    Plans a reach motion to a target point: solves inverse kinematics,
    generates a smooth trajectory, and checks whether the arm's real
    motors can actually execute that motion. Requires payload_mass_kg > 0.
    """
    config_errors = validate_config(config)
    if config_errors:
        raise HTTPException(status_code=400, detail={"config_errors": config_errors})

    try:
        result = plan_reach_motion(
            config,
            target_position=(target_x, target_y, target_z),
            tracking_tolerance_rad=tracking_tolerance_rad,
            record_trajectory=record_trajectory,
        )
        return result
    except IKError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Motion planning failed: {exc}")



# Mounted last and deliberately: Starlette matches routes in registration
# order, so every /health, /demo/..., /analyze/arm route above is checked
# first. Only requests that don't match any of those fall through to serving
# the frontend static files (index.html at "/", etc.).
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
