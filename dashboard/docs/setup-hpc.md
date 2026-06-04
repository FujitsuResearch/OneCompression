# HPC cluster (SLURM) setup

How to run Backend, Worker, Redis, and vLLM on a SLURM-managed GPU node
without Docker, plus troubleshooting.

## Prerequisites

| Item | Value |
|---|---|
| Job scheduler | SLURM (`salloc` / `sbatch`) |
| GPU node example | `gpu08` |
| Python | 3.12 (system or uv-managed) |
| Package manager | uv |
| CUDA | 12.x / 13.x (`pyproject.toml` uses `pytorch-cu130`) |
| Node.js (local PC) | 20 LTS or newer (18+; Vite 6 / esbuild require Node 18+) |

## 1. First-time setup

### 1.1 Backend venv

```bash
cd backend/
uv sync          # creates .venv from pyproject.toml
```

### 1.2 vLLM (bundled in the main venv)

The current `onecomp` and `vllm` (0.21.x) live in **the same venv**
(`backend/.venv`). `uv sync` installs `onecomp`, `torch` (PyTorch cu130),
and `vllm` together.

vLLM is still launched **as a separate process** from the Celery worker,
but the interpreter stays in the main venv:

```bash
# default in config.py (rarely needs changing)
vllm_python: str = ".venv/bin/python"
```

After editing `config.py`, **restart the worker** (`--reload` on the API
does not propagate).

> Since OneCompression v1.1.1, vLLM 0.20.x and later coexist with `onecomp`
> in the same venv, so a separated venv is usually unnecessary.
> Use the separation procedure in **1.2.1** only if `uv sync` fails.

> `uv tool install redis-server` has no Linux wheel and fails. Use the
> source build in **1.3** for Redis.

### 1.2.1 Separate vLLM Python venv (optional)

When `uv sync` cannot resolve `onecomp` and `vllm` together, or when vLLM
requires a different PyTorch / CUDA wheel, split the environments into
a **quantization venv** (`.venv`) and a **vLLM venv** (`.venv-vllm`).

| venv | Role | Started by |
|---|---|---|
| `backend/.venv` | onecomp quantization, FastAPI, Celery worker | `start_worker.py` / `start_backend.py` |
| `backend/.venv-vllm` | vLLM OpenAI API server only | Worker via `subprocess` |

The app code never imports vLLM directly. `deploy_vllm()` only spawns a
separate process via `settings.vllm_python`, so separating venvs has no
ripple effect on the rest of the code.

**1. Create the separated venv**

```bash
cd backend/
bash setup_vllm.sh
```

Adjust the PyTorch index inside `setup_vllm.sh` to match the cluster's
CUDA (e.g. `cu130` for CUDA 13.x):

```bash
uv pip install --python "${VENV_DIR}/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cu130
```

**2. Edit `config.py`**

Point `vllm_python` in `backend/app/core/config.py` at the separated venv
(**the relative path assumes the worker is started with `backend/` as its
cwd**):

```python
    vllm_python: str = ".venv-vllm/bin/python"
```


**3. Override via environment variable (when you do not want to edit `config.py`)**

```bash
cd backend/
export ONECOMP_VLLM_PYTHON="$(pwd)/.venv-vllm/bin/python"
```

Environment variables with the `ONECOMP_` prefix take precedence over
`config.py` defaults.

**4. Restart the worker (required)**

```bash
pkill -f start_worker.py
cd backend && . .venv/bin/activate
export ONECOMP_DEVICE=cuda
# also set this if you did not edit config.py
export ONECOMP_VLLM_PYTHON="$(pwd)/.venv-vllm/bin/python"
nohup uv run python start_worker.py > /tmp/worker.log 2>&1 &
```

**5. Verify**

```bash
cd backend/
.venv/bin/python -c "from app.core.config import settings; print(settings.vllm_python)"
test -x .venv-vllm/bin/python && .venv-vllm/bin/python -c "import vllm; print(vllm.__version__)"
```

After **Stop → Deploy** in the UI, `/tmp/worker.log` should contain
`Starting vLLM: .../.venv-vllm/bin/python ...`.

**Removing vLLM from the main venv (optional)**

To avoid dependency conflicts in `uv sync`, you can also delete
`vllm>=0.21.0` from `pyproject.toml` and re-run `uv sync`. Quantization
and the API then live in `.venv` only; the inference server lives in
`.venv-vllm` only.

### 1.3 Redis (local build)

`apt install redis` is often unavailable on HPC nodes, so compile Redis
from source and run it in user space.

```bash
cd backend/
wget https://github.com/redis/redis/archive/refs/tags/7.2.7.tar.gz
tar xzf 7.2.7.tar.gz
cd redis-7.2.7
make -j$(nproc)
```

## 2. Startup procedure

### 2.1 Allocate a GPU node

```bash
salloc -p interactive --time=04:00:00 --gres=gpu:1
```

Once a node is allocated (e.g. `gpu08`), run every command below
**on that node**.

### 2.2 Start Redis

```bash
cd backend/
./redis-7.2.7/src/redis-server --daemonize yes --bind 127.0.0.1
```

> **Important**: always pass `--bind 127.0.0.1` (see Troubleshooting below
> for the reason).

### 2.3 Start the worker

```bash
cd backend/
. .venv/bin/activate
export ONECOMP_DEVICE=cuda
# On nodes without nvcc (typical HPC); see Troubleshooting #10
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup uv run python start_worker.py > /tmp/worker.log 2>&1 &
```

### 2.4 Start the backend API

```bash
ONECOMP_DEVICE=cuda python start_backend.py --reload --port 8001
```

Binding to port 8001 lets the frontend reach the GPU-node API.

> If port 8000 is already in use you will see `Address already in use`.
> Choose another port such as `--port 8001`.

### 2.5 Two terminals on the same salloc job

To run the worker (background) and API (foreground) in separate terminals,
attach to the same job from the login node:

```bash
squeue -u $USER                    # find the JOBID (e.g. 41229)
srun --jobid=41229 --pty bash      # open a shell on the GPU node
cd ~/onecomp-lab/dashboard/backend
. .venv/bin/activate
```

First terminal: worker / Redis. Second terminal: backend API.

### 2.6 Access from a local PC (SSH tunnel)

The API listens on the **GPU node allocated by `salloc`** (e.g. `gpu08`).
`localhost:8001` on the login node does **not** reach it.

```
local PC --SSH--> login01:8001 --???--> gpuAA:8001 (FastAPI)  ← this hop is the gotcha
```

**Recommended**: point `LocalForward` in your local `~/.ssh/config` at the
GPU node name.

```
Host XXX01
    HostName XXX01
    LocalForward 8001 gpuAA:8001
    ServerAliveInterval 60
```

- `gpuAA` changes on every `salloc`. Read it from `squeue` or the `salloc`
  output and update the config each time.
- Leaving `LocalForward 8001 localhost:8001` makes the tunnel terminate at
  the login node and you may see
  `channel open failed: connect failed: Connection refused`.

`LocalForward` in `~/.ssh/config` only defines the rule; it does **not**
start a tunnel by itself. On the local PC, open a dedicated terminal and keep
an SSH session to the login node while you use the UI:

```bash
ssh XXX01 -N
```

- **`-N`**: no remote shell — port forwarding only (leave this terminal open).
- **`-f`** (optional): run in the background (`ssh -f XXX01 -N`).
- A normal `ssh XXX01` session also forwards ports, but closing that shell
  drops the tunnel.

The backend API must already be listening on the GPU node
(`start_backend.py --port 8001`, **§2.4**) before `curl` or the frontend can
reach it.

Connectivity check (with `ssh XXX01 -N` running):

```bash
curl http://localhost:8001/api/health   # on the local PC. {"status":"ok"} means OK
```

### 2.7 Frontend

The Vite dev server runs on your **local PC** (not on the GPU node). Install
Node.js there first, then install frontend dependencies once per clone.

**Install Node.js (local PC)**

| OS | Typical install |
|---|---|
| Windows | [nodejs.org](https://nodejs.org/) LTS installer, or `winget install OpenJS.NodeJS.LTS` |
| macOS | [nodejs.org](https://nodejs.org/) LTS, or `brew install node` |
| Linux | [nodejs.org](https://nodejs.org/) binary / distro packages, or [nvm](https://github.com/nvm-sh/nvm) (`nvm install --lts`) |

`npm` is bundled with Node.js. **Yarn** is optional (`npm install -g yarn` if
you prefer `yarn` over `npm`).

Verify:

```bash
node -v    # v20.x or v22.x recommended
npm -v
```

**Install dependencies** (from the `dashboard/` directory):

```bash
cd frontend/
npm install
# or: yarn
```

When running Vite locally, point the API target at the SSH tunnel
(**§2.6**).

Unix shell (bash, zsh, Git Bash on Windows):

```bash
cd frontend/
VITE_API_TARGET=http://localhost:8001 yarn dev
# or
VITE_API_TARGET=http://localhost:8001 npm run dev
```

Windows PowerShell:

```powershell
cd frontend
$env:VITE_API_TARGET = "http://localhost:8001"
npm run dev
# or: yarn dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

### 2.8 Which inference path is in use (vLLM vs onecomp)

vLLM is used only when `ONECOMP_DEVICE=cuda` **and** vLLM deployment
succeeded.

| Condition | Deploy | Chat API behavior |
|---|---|---|
| `ONECOMP_DEVICE=cuda` + vLLM started successfully | `deploy_vllm` | `POST /chat` returns the message **immediately** (`inference_url` set) |
| Otherwise / vLLM failed | `deploy_onecomp` | Returns a `task_id`; the client **polls** `GET /api/jobs/chat-result/{task_id}` |

Check the state in the DB:

```bash
cd backend/
.venv/bin/python -c "
from app.core.database import SessionLocal
from app.models.job import Job
db = SessionLocal()
j = db.get(Job, '<job-id>')
print('inference_url:', j.inference_url)
print('inference_pid:', j.inference_pid)
db.close()
"
```

- `inference_url` still `None` → still on onecomp direct inference. To
  switch to vLLM, **Stop → Deploy** again from the UI.
- Chat keeps polling `chat-result` → not on the vLLM path.

### 2.9 Stopping processes

```bash
pkill -f start_worker.py              # Celery worker
pkill -f "vllm.entrypoints"           # vLLM
fuser -k 8090/tcp                     # if the vLLM port is stuck
```

After changing any setting, **restart the worker** or it will keep running
old code.

```bash
pkill -f start_worker.py
ONECOMP_DEVICE=cuda nohup uv run python start_worker.py > /tmp/worker.log 2>&1 &
tail -f /tmp/worker.log               # look for "Starting vLLM" on the next deploy
```

## 3. Application architecture

In HPC operation Docker, PostgreSQL, and MinIO are not used. 
Everything runs as processes on the GPU node.

### 3.1 Database and model storage

| Item | Location | Setting |
|---|---|---|
| Job metadata | `backend/onecomp.db` (SQLite) | `ONECOMP_DATABASE_URL` (default `sqlite:///./onecomp.db`) |
| Quantized models | `backend/tmp/quantized/<job-id>/` | `ONECOMP_QUANTIZED_DIR` (default `tmp/quantized`) |

`database.py` sets `check_same_thread=False` on SQLite so that FastAPI
and Celery can share the same file.

### 3.2 Task queue (Redis + Celery)

- Redis runs from the binary built in **1.3** on the GPU node
- The default URL is `redis://127.0.0.1:6379/0` (**`127.0.0.1` required**;
  `localhost` may fail to resolve over IPv6 → see **#1**)
- The Celery worker uses `--pool=solo` (GPU tasks run serially in a
  single process)

### 3.3 Python environment

| Package | How to install |
|---|---|
| `onecomp`, `vllm`, `torch`, etc. | `cd backend && uv sync` |
| PyTorch | `pytorch-cu130` index in `pyproject.toml` (for CUDA 13.x drivers) |

Quantization runs inside the worker (main `.venv`). On CUDA inference,
**Deploy** spawns vLLM as a **separate subprocess** via
`settings.vllm_python` (default `.venv/bin/python`, see **1.2**).

### 3.4 Pipeline

1. **Quantization** — the `run_quantization` task quantizes on the GPU
   with `onecomp` and saves to `tmp/quantized/<job-id>/`
2. **Deploy** — the `deploy_model` task calls `deploy_vllm()` when
   `ONECOMP_DEVICE=cuda`, otherwise `deploy_onecomp()`
3. **Chat** — if vLLM is running (`inference_url` set), the API HTTP-proxies
   to vLLM. Otherwise it falls back to onecomp inference via Celery (polled)

### 3.5 Implementation notes (code)

`app/services/inference.py` `deploy_vllm()` does:

- Kills any prior vLLM process before redeploying (`_stop_existing_deployment`)
- Computes `--gpu-memory-utilization` from free VRAM
  (`_pick_gpu_memory_utilization`)
- Calls `torch.cuda.empty_cache()` before deploy, passes `--enforce-eager`
  among other options
- On failure, copies error lines from `/tmp/vllm-<job-id>.log` into the
  job's `error_message`

`app/worker/tasks.py` releases the CUDA cache after quantization to prevent
OOM at deploy time.

To work around HPC HTTP proxies, `inference.py` appends
`localhost,127.0.0.1` to `no_proxy` / `NO_PROXY` at import time (see **#2**).

## 4. Environment variables

All use the `ONECOMP_` prefix. Defaults live in
`backend/app/core/config.py` and can be overridden via env vars.

| Variable | Default | Description |
|---|---|---|
| `ONECOMP_DATABASE_URL` | `sqlite:///./onecomp.db` | DB URL (path relative to `backend/`) |
| `ONECOMP_REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis URL |
| `ONECOMP_DEVICE` | `cpu` | Execution device. **`cuda` on HPC** (both worker and API) |
| `ONECOMP_MOCK_QUANTIZATION` | `false` | `true` skips quantization (smoke test) |
| `ONECOMP_WORKER_HOST` | `localhost` | Hostname used in the vLLM `inference_url` |
| `ONECOMP_VLLM_PORT` | `8090` | vLLM listen port |
| `ONECOMP_VLLM_PYTHON` | `.venv/bin/python` | Python for the vLLM subprocess (see **1.2.1** when separated) |
| `ONECOMP_QUANTIZED_DIR` | `tmp/quantized` | Where quantized models are stored |
| `ONECOMP_CHAT_TIMEOUT` | `900` | Chat HTTP timeout in seconds |
| `VLLM_USE_FLASHINFER_SAMPLER` | (vLLM default) | Set to `0` on the **worker** when `nvcc` is missing ([#10](#10-vllm-deploy-fails--could-not-find-nvcc)) |

## 5. Troubleshooting

### #1: Redis connection error — `Address family not supported by protocol` (Errno 97)

**Symptom**:

```
redis.exceptions.ConnectionError: Error 97 connecting to localhost:6379.
Address family not supported by protocol.
```

and on the Celery side:

```
RuntimeError:
Retry limit exceeded while trying to reconnect to the Celery result store
backend. The Celery application must be restarted.
```

**Cause**:

The kernel on the HPC node (gpu08) has IPv6 disabled.
When `localhost` is resolved to `::1` (IPv6) via `/etc/hosts`,
Python's `socket.socket()` tries to create an IPv6 socket and fails with
`EAFNOSUPPORT` (errno 97).

Celery (Redis backend) keeps retrying, hits the retry limit, raises
`RuntimeError: Retry limit exceeded`, and the entire application stops
responding.

**Fix**:

1. Force IPv4 by putting `127.0.0.1` in the Redis URL:

```python
redis_url: str = "redis://127.0.0.1:6379/0"
```

2. Bind the Redis server to `127.0.0.1` as well:

```bash
./redis-7.2.7/src/redis-server --daemonize yes --bind 127.0.0.1
```

### #2: vLLM health check is routed through the HTTP proxy

**Symptom**:

vLLM's `/health` endpoint keeps returning `403 Forbidden`.
The response body contains
`Copyright (C) 1996-2022 The Squid Software Foundation`.

**Cause**:

HPC environments often have `http_proxy` / `https_proxy` set, so
`httpx.get("http://localhost:8090/health")` is routed through the proxy
(Squid).

**Fix**:

Set `no_proxy` automatically when `inference.py` is imported:

```python
for _var in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_var, "")
    if "localhost" not in _cur:
        os.environ[_var] = f"{_cur},localhost,127.0.0.1" if _cur else "localhost,127.0.0.1"
```

### #3: CUDA OOM at vLLM deploy time

**Symptom**:

Deploying vLLM right after quantization fails with CUDA out-of-memory.

**Cause**:

Quantization holds onto GPU memory that is not released, so vLLM cannot
allocate enough free memory.

**Fix**:

1. Call `gc.collect()` + `torch.cuda.empty_cache()` after the quantization task finishes
2. Clear the CUDA cache again right before deploying vLLM
3. Compute `--gpu-memory-utilization` from actual free VRAM in
   `_pick_gpu_memory_utilization()`
4. Disable CUDA graphs with `--enforce-eager` to lower memory usage

### #4: vLLM redeploy fails

**Symptom**:

A second deploy fails with `Address already in use`, or the previous
process is left as a zombie.

**Fix**:

`_stop_existing_deployment()` runs before deploy and:

1. Looks up jobs with an `inference_pid` in the DB and `SIGKILL`s them
2. Calls `pkill -9 -f vllm.entrypoints.openai.api_server` to kill any
   stray process outside the process group
3. Waits up to 10 seconds for the port to be released

### #5: SSH tunnel is up but the API is unreachable

**Symptom**:

- SSH from the local PC connects successfully
- `channel 3: open failed: connect failed: Connection refused` shows up,
  or the API never responds

**Cause**:

`LocalForward 8001 localhost:8001` forwards to port 8001 on the **login
node**, but FastAPI is listening on the **GPU node** (`salloc` target).

**Fix**:

In `~/.ssh/config`, set `LocalForward` to the GPU node name such as
`gpu08:8001` (see **2.6**).

### #6: Chat is slow / vLLM is not being used

**Symptom**:

- Responses take about 6 seconds
- Server logs keep showing `GET /api/jobs/chat-result/...` polling
- `POST /chat` does not return the body immediately

**Cause**:

1. `ONECOMP_DEVICE=cuda` is unset, so worker / API are in CPU mode
2. The job was created before vLLM was deployed, so `inference_url` is
   still `None` (onecomp direct inference)
3. vLLM failed to start (OOM, proxy 403, stale process holding the port),
   so the system is effectively in `deploy_onecomp` fallback

**Fix**:

1. Restart worker / API with `ONECOMP_DEVICE=cuda`
2. **Stop → Deploy** from the UI
3. Check in the DB whether `inference_url` is set to something like
   `http://localhost:8090` (see **2.8**)
4. `tail -f /tmp/worker.log` for `Starting vLLM` / `vLLM ready`.
   On failure, inspect `/tmp/vllm-<job-id>.log`

### #7: Deploy fails — `vllm_python` path not found

**Symptom**:

```
[Errno 2] No such file or directory: '.venv-vllm/bin/python'
```

or

```
[Errno 2] No such file or directory: '.venv/bin/python'
```

**Causes (most common first)**:

1. **`config.py` / env var disagrees with the actual venv layout** — e.g.
   `vllm_python` is still `.venv-vllm/bin/python` but `.venv-vllm` was
   never created, or vice versa
2. **Celery worker was not restarted** — `config.py` was edited but the
   running worker still holds the old `settings`
3. **Worker cwd is not `backend/`** — the relative path
   `.venv/bin/python` cannot be resolved

**Fix**:

| venv layout | `vllm_python` in `config.py` | Prereq |
|---|---|---|
| Main only (typical) | `.venv/bin/python` | `uv sync` only |
| Separated (see #9) | `.venv-vllm/bin/python` or absolute path | `bash setup_vllm.sh` + **#9** |

Either way, restart the worker:

```bash
pkill -f start_worker.py
cd backend && . .venv/bin/activate
export ONECOMP_DEVICE=cuda
# only when using a separated venv:
# export ONECOMP_VLLM_PYTHON="$(pwd)/.venv-vllm/bin/python"
nohup uv run python start_worker.py > /tmp/worker.log 2>&1 &
```

Then **Stop → Deploy** from the UI. If `error_message` in the DB changes,
the new settings are being applied.

### #10: vLLM deploy fails — `Could not find nvcc`

**Symptom**:

- Quantization completes (`status=completed`)
- **Deploy & Start Chat** fails; the Chat tab shows a deploy error
- Job `error_message` (and `/tmp/vllm-<job-id>.log`) contain:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

**Cause**:

vLLM may use the FlashInfer sampler, whose initialization runs a JIT build
that requires `nvcc` (CUDA toolkit). HPC GPU nodes often have a working
CUDA **driver/runtime** (enough for `ONECOMP_DEVICE=cuda` quantization) but
no toolkit install, so vLLM engine startup fails even though the quantized
model is valid.

**Fix**:

1. Set the variable on the **Celery worker** process (not the FastAPI API):

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

2. **Restart the worker** (the API's `--reload` does not affect Celery):

```bash
pkill -f start_worker.py
cd backend && . .venv/bin/activate
export ONECOMP_DEVICE=cuda
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup uv run python start_worker.py > /tmp/worker.log 2>&1 &
```

3. In the UI: **Stop → Deploy** (or **Deploy & Start Chat** again)

**Notes**:

- Setting `VLLM_USE_FLASHINFER_SAMPLER` only on the API or in the browser
  has **no effect** — vLLM is spawned from the worker via `deploy_vllm()`.
  Export the variable in the same shell session (or job script) that starts
  `start_worker.py`; the vLLM child process inherits the worker environment.
- If deploy still fails, inspect `tail -80 /tmp/vllm-<job-id>.log` and
  `tail -f /tmp/worker.log`.

### #8: Celery worker crashes immediately (`setproctitle`)

**Symptom**:

```
AttributeError: module 'setproctitle' has no attribute 'setproctitle'
```

**Fix**:

```bash
cd backend/
uv sync    # setproctitle is in pyproject.toml
```

### #9: Separating onecomp and vllm — `vllm_python` setup

**When to separate**

- `uv sync` cannot resolve `onecomp` and `vllm` together (e.g. on
  `transformers`)
- vLLM needs a different PyTorch / CUDA build (e.g. `cu128` vs `cu130`)
- vLLM import / startup fails in the main venv during deploy, but
  `import vllm` works in a separated venv

**Procedure (summary; details in 1.2.1)**:

1. `cd backend && bash setup_vllm.sh` (change the CUDA index to `cu130`
   etc. if needed)
2. Edit `backend/app/core/config.py`:

```python
class Settings(BaseSettings):
    # ...
    vllm_python: str = ".venv-vllm/bin/python"
```

3. Or skip editing `config.py` and set the env var at worker start:

```bash
export ONECOMP_VLLM_PYTHON="$(pwd)/.venv-vllm/bin/python"
```

4. `pkill -f start_worker.py`, then restart the worker with
   `ONECOMP_DEVICE=cuda`
5. **Stop → Deploy**. Verify `tail /tmp/worker.log` shows
   `.venv-vllm/bin/python` in the command line

**Notes when editing `config.py`**

- Change only the **single line** `vllm_python` (leave `device`,
  `redis_url`, etc. alone)
- If you set the default to `.venv-vllm/...` but never create
  `.venv-vllm`, you get the same Errno 2 as in #7
- To go back to main-venv-only operation, set
  `vllm_python: str = ".venv/bin/python"`
- Changes only take effect after a **worker restart**
  (FastAPI's `--reload` does not affect the worker)

**Optional: avoid the dependency conflict altogether**

Removing `vllm>=0.21.0` from `pyproject.toml` and re-running `uv sync`
leaves the main venv with `onecomp` only and the `.venv-vllm` with vLLM
only.
