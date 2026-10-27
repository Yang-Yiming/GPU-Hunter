# gpu-hunter

Watch NVIDIA GPUs and automatically start your job as soon as enough cards are **stably idle**.

Two ways to use it:

1. **Single job**: one command plus a GPU count and preferences; run as soon as the cards are free (the classic case).
2. **Workflow**: a YAML file defines a multi-job queue, run concurrently as slots open up (greedy), or strictly in order (sequential).

## Installation

```bash
pip install -e .        # dependencies: nvidia-ml-py, PyYAML; Linux + NVIDIA driver
```

## Quick start

### Single-job mode

```bash
# Wait for 1 completely idle GPU, then run train.py
gpu-hunter -- python train.py

# Need 2 GPUs, prefer GPU2 and GPU3
gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py --epochs 100

# Each GPU needs 40GB free (does not have to be fully idle, but must have no compute processes)
gpu-hunter --mem-per-gpu 40G -- bash run.sh
```

A lone `xx.py` / `xx.sh` gets an interpreter automatically; everything after `--` is executed as-is as the child command.

### Workflow mode

```bash
gpu-hunter workflow.yaml
```

```yaml
# workflow.yaml
defaults:                # global defaults; any job can override them per field
  mem-per-gpu: 40G

schedule: greedy         # greedy (default): fill free slots | sequential: strict listed order

interval: 5              # polling tick in seconds
stable-rounds: 3         # consecutive idle rounds required before acting
grace: 45                # grace period in seconds for launch confirmation

jobs:
  - name: pretrain       # name is used in logs and the summary; defaults to job-1..N
    cmd: python train.py --epochs 100   # strings are shlex-split; a list [python, ...] also works
    gpu-num: 4           # how many GPUs must be free at the same time
    prefer: [0, 1]       # ranking preference (soft: fall back to other cards)

  - name: eval-a
    cmd: bash eval.sh --dataset a
    allow: [7]           # allow-list (hard constraint: wait rather than use another card)
    env: { DEBUG: "1" }  # extra environment variables
    log: logs/eval_a.log # default is logs/<name>-<timestamp>.log; "-" streams to the terminal

  - name: quick-check
    cmd: ./check.sh
    gpu-num: 1           # omitted fields inherit from defaults
```

#### Scheduling semantics

| Mode | Behavior |
|---|---|
| `greedy` (default) | Every tick scans all pending jobs: **jobs that do not fit are skipped and scanning continues** (big jobs do not block small ones). Multiple jobs may launch in the same tick when resources allow, each taking disjoint GPUs. |
| `sequential` | Only the head of the queue is considered; the next job starts after it finishes (including on failure). |

Failure handling: failures are recorded and the queue continues; a summary table is printed when the workflow ends, and **the exit code is the number of failed jobs** (0 when all succeed). A preempted job is rolled back and requeued; the `try` column shows the attempt count.

## Single-job options

| Option | Default | Description |
|---|---|---|
| `--gpu-num N` | 1 | How many GPUs must be free at the same time |
| `--mem-per-gpu SIZE` | - | Free memory required per GPU, e.g. `40G`; default requires a fully idle card |
| `--prefer LIST` | - | Soft preference: `2,3` tries 2 first, then 3, falling back to other cards |
| `--allow LIST` | all | Hard allow-list: only pick from the listed GPUs |
| `--interval SEC` / `--stable-rounds K` | 5 / 3 | Polling interval / stability window |
| `--idle-mem SIZE` | 500M | Usage below this counts as idle (also requires utilization < 5% and no compute processes) |
| `--grace SEC` / `--heartbeat SEC` | 45 / 120 | Launch confirmation grace period / runtime heartbeat |
| `--log FILE` / `--cwd DIR` | terminal / current directory | Redirect job output / child working directory |

Difference between `prefer` and `allow`:

```
all GPUs -> allow filter (hard) -> stable-idle check -> prefer ranking (soft) -> take the first N
```

- `--prefer 3,2`: 3 is preferred, then 2; if neither is free, use other cards.
- `--allow 2,3`: only 2 and 3 are allowed; other cards stay untouched even when free.
- Multiple GPUs are taken greedily in preference order: `--prefer 3 --gpu-num 2` with {0,2,3} idle -> {3,0}.

## How it works

1. **Probing**: uses NVML (the library behind `nvidia-smi`) and never parses text output.
2. **Stability window**: acts only after K consecutive idle polls in a sliding window; multi-GPU requests are filled atomically.
3. **Launch**: injects `CUDA_VISIBLE_DEVICES` and starts the child in an independent session; inside a workflow, a resource ledger keeps concurrent jobs on disjoint GPUs.
4. **Claim verification**: checks for compute processes on the target GPUs during the grace period. Seeing our own pid clears the claim (torchrun workers are not mistaken for a preemption); a foreign process that persists while our pid never appears triggers a rollback and requeue; early child exit ends verification immediately; invisible pids inside containers downgrade to a warning and continue.
5. **Signals**: SIGINT/SIGTERM are forwarded to every child process group; two presses within 5 seconds escalate to SIGKILL; Ctrl-C also tears down every job in a workflow.
6. **Single-instance lock**: flock ensures only one hunter runs per machine.

## Running in the background

```bash
nohup gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py &
nohup gpu-hunter workflow.yaml > wf.out 2>&1 &   # job output goes to the per-job log files
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # no real GPU required
```
