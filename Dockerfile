FROM python:3.12-slim

# libgl1: required by cadquery/OCP (OpenCASCADE bindings) - discovered
# during local dependency testing; without it, `import cadquery` fails
# with "libGL.so.1: cannot open shared object file".
# build-essential (g++): pybullet has no prebuilt wheel for this
# base image's platform/Python ABI, so pip falls back to compiling it
# from source - discovered during the first local Docker build attempt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (not the whole app) so Docker's layer cache
# only re-runs the slow pip install when dependencies actually change,
# not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Local file storage under data/ - TEMPORARY, will be replaced by real
# database + object storage in the next productionization phase. For
# now this just needs to exist so the app doesn't fail on startup.
RUN mkdir -p data/meshes data/models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
