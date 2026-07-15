# syntax=docker/dockerfile:1
FROM python:3.12-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    LIBERO_CONFIG_PATH=/opt/libero-config \
    MUJOCO_GL=egl \
    PYTHONPATH=/opt/LIBERO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0t64 \
        libglew2.2 \
        libglfw3 \
        libgomp1 \
        libice6 \
        libosmesa6 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ARG LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
RUN git init /opt/LIBERO \
    && cd /opt/LIBERO \
    && git remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    && git fetch --depth 1 origin "${LIBERO_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && rm -rf /opt/LIBERO/.git

COPY vlm_benchmarking/requirements-demo.txt /tmp/requirements-demo.txt
COPY vlm_benchmarking/constraints-demo.txt /tmp/constraints-demo.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==26.1.2 setuptools==83.0.0 wheel==0.47.0 \
    && python -m pip install --constraint /tmp/constraints-demo.txt \
        --requirement /tmp/requirements-demo.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --editable /opt/LIBERO \
    && python -m pip check \
    && mkdir -p /opt/libero-config /opt/LIBERO/libero/datasets \
    && printf '%s\n' \
        'assets: /opt/LIBERO/libero/libero/assets' \
        'bddl_files: /opt/LIBERO/libero/libero/bddl_files' \
        'benchmark_root: /opt/LIBERO/libero/libero' \
        'datasets: /opt/LIBERO/libero/datasets' \
        'init_states: /opt/LIBERO/libero/libero/init_files' \
        > /opt/libero-config/config.yaml \
    && python -c "from libero.libero import get_libero_path; from libero.libero.envs import OffScreenRenderEnv; assert get_libero_path('bddl_files')"

WORKDIR /app/vlm_benchmarking
COPY vlm_benchmarking/demo ./demo
COPY vlm_benchmarking/vlm_bench ./vlm_bench
RUN python -m demo.deployment_smoke

EXPOSE 7860

CMD ["python", "-u", "-m", "demo", "--input-dir", "demo/libero_demo_cache", "--output-dir", "demo/libero_prediction_cache", "--so101-config", "demo/so101_demo_cache/config.yaml", "--host", "0.0.0.0", "--port", "7860", "--res", "1024", "--no-disk-cache", "--no-open-browser"]
