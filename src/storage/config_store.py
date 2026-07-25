"""
Simple file-based persistence for arm designs - one JSON file per saved
design under data/saved_designs/. No database, no server-side state beyond
the filesystem: appropriate for a solo, local-first tool.

Only the ArmConfig (the design) is saved - not analysis results. Torque
checks and lift tests are always recomputed live from the current code,
so a saved design never goes stale relative to bug fixes or model changes
in the analysis itself.

SECURITY NOTE: config.name is user-supplied and is used to build a file
path. Without sanitization, a name like "../../etc/passwd" could write
or read outside the intended directory. _slugify() strips everything
except alphanumerics, underscore, and hyphen specifically to prevent this -
don't relax that pattern without re-checking path traversal.
"""

import json
import re
import datetime
from pathlib import Path

from src.urdf_generator.schema import ArmConfig

STORAGE_DIR = Path("data/saved_designs")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    slug = slug.strip("_") or "unnamed"
    return slug[:100]


def _path_for(name: str) -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR / f"{_slugify(name)}.json"


def save_design(config: ArmConfig) -> dict:
    slug = _slugify(config.name)
    path = _path_for(config.name)
    saved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    payload = {
        "config": config.model_dump(),
        "saved_at": saved_at,
    }
    path.write_text(json.dumps(payload, indent=2))

    return {"name": config.name, "slug": slug, "saved_at": saved_at}


def list_designs() -> list[dict]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    designs = []
    for path in STORAGE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            designs.append({
                "name": data["config"]["name"],
                "slug": path.stem,
                "saved_at": data.get("saved_at"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    designs.sort(key=lambda d: d["saved_at"] or "", reverse=True)
    return designs


def load_design(slug: str) -> ArmConfig:
    safe_slug = _slugify(slug)
    path = STORAGE_DIR / f"{safe_slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved design found for '{slug}'")
    data = json.loads(path.read_text())
    return ArmConfig(**data["config"])


def delete_design(slug: str) -> bool:
    safe_slug = _slugify(slug)
    path = STORAGE_DIR / f"{safe_slug}.json"
    if path.exists():
        path.unlink()
        return True
    return False
