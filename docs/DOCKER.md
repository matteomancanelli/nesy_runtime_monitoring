# Running the experiments on the GPU server (Docker)

Everything the experiments need — Python, the `nesy-monitoring` conda env,
torch+CUDA, and the **MONA** binary that `ltlf2dfa` depends on — is baked into
a single Docker image. You do **not** install anything on the server host.

> **Why this satisfies "I can't install anything on the server":**
> The `apt-get install mona` line in the `Dockerfile` runs *inside the image at
> build time*, in the container's own filesystem. It never modifies the host
> and needs no host privileges. The only thing the host must already provide is
> the **NVIDIA driver + nvidia-container-toolkit** (standard on a GPU server
> that asked you to use Docker) — that's what `--gpus all` talks to. The torch
> cu124 wheel ships its own CUDA runtime, so no CUDA install is needed either.

---

## Option A — build on the server (server has internet + `docker build`)

```bash
# from the repo root, on the server
docker build -t nesy-monitoring .
```

Then run an experiment, mounting `results/` so output lands on the host and
survives restarts:

```bash
docker run --rm --gpus all \
    -v "$PWD/results:/app/results" \
    nesy-monitoring \
    python experiments/exp1_single_trace.py
```

Run all three back-to-back:

```bash
docker run --rm --gpus all \
    -v "$PWD/results:/app/results" \
    nesy-monitoring \
    bash scripts/run_all.sh
```

Long runs — detach and keep logs:

```bash
docker run -d --name exp2 --gpus all \
    -v "$PWD/results:/app/results" \
    nesy-monitoring \
    python experiments/exp2_formula_complexity.py
docker logs -f exp2
```

## Option B — build locally, ship the image (server can't build / is offline)

Build here (this machine has internet, conda, and MONA), then transfer the
image as a file — no registry needed:

```bash
# on this machine
docker build -t nesy-monitoring .
docker save nesy-monitoring | gzip > nesy-monitoring.tar.gz
scp nesy-monitoring.tar.gz user@server:~/
```

```bash
# on the server
docker load < nesy-monitoring.tar.gz
docker run --rm --gpus all -v "$PWD/results:/app/results" \
    nesy-monitoring python experiments/exp1_single_trace.py
```

(You still need the repo on the server for the mounted `results/` dir, or just
mount any writable host directory in its place.)

## docker compose (optional)

If the server's Docker Compose is v2.30+, `docker-compose.yml` wraps the flags:

```bash
docker compose build
docker compose run --rm exp python experiments/exp1_single_trace.py
docker compose run --rm exp bash scripts/run_all.sh
```

---

## Resuming after an interruption

Results are written **incrementally**: each (monitor, x-value) cell is appended
to its `results/*.csv` the moment it finishes. If a run is killed (OOM, timeout,
disconnect), just launch it again — it reads the existing CSV, skips every cell
already present, and continues. The plot is regenerated from the full CSV at the
end of each run.

To force a clean recompute, delete the relevant CSV first:

```bash
rm results/exp2_formula_complexity.csv
```

> **Note for *local* reruns:** the `results/*.csv` currently checked in predate
> the `device` column and have a different schema, so don't append to them —
> delete or move them before resuming locally. On the server this never arises:
> the mounted `results/` starts empty (it's excluded from the image).

## Verifying the image (optional)

```bash
# the test suite (expect 293 passed, 6 xfailed)
docker run --rm --gpus all nesy-monitoring pytest -q

# confirm the GPU is visible inside the container
docker run --rm --gpus all nesy-monitoring \
    python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, the host is missing
nvidia-container-toolkit or you omitted `--gpus all`. The experiments still run
(they fall back to CPU via `DEVICE = 'cuda' if ... else 'cpu'`), just without
the batched-GPU advantage Exp 3 is meant to show.
