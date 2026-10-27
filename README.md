# gpu-hunter

Watch NVIDIA GPUs and automatically start your job as soon as enough cards are **stably idle**.

Three ways to use it:

1. **Single job**: one command plus a GPU count and preferences; run as soon as the cards are free (the classic case).
2. **Workflow**: a YAML file defines a multi-job queue, run concurrently as slots open up (greedy), or strictly in order (sequential).
3. **Reserve (hold)**: hold N GPUs x M memory with placeholder processes, then hand them to whatever you want later -- useful when the real task cannot be described as a command (interactive work, a GUI, a manual sequence of steps).

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

# Each GPU needs 40GB free (no compute processes; utilization must also be <= 5%)
gpu-hunter --mem-per-gpu 40G -- bash run.sh
```

A lone `xx.py` / `xx.sh` gets an interpreter automatically; everything after `--` is executed as-is as the child command.

By default the wait loop is quiet: each failed polling attempt prints a single `x` without a newline, while a match, launch, warning or job completion prints a normal text line. Pass `-v` / `--verbose` to get the full timestamped status log.

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
idle-mem: 500            # used memory below this counts as idle
idle-util: 5             # utilization at or below this counts as idle
grace: 45                # grace period in seconds for launch confirmation
claim-stable: 8          # a foreign process must persist this long to count as preemption
max-attempts: 3          # preempted jobs are retried at most this many times
retry-delay: 2           # base retry delay in seconds (doubles on each retry)

jobs:
  - name: pretrain       # name is used in logs and the summary; defaults to job-1..N
    cmd: python train.py --epochs 100   # strings are shlex-split; a list [python, ...] also works
    gpu-num: 4           # how many GPUs must be free at the same time
    prefer: [0, 1]       # ranking preference (soft: fall back to other cards)

  - name: eval-a
    cmd: bash eval.sh --dataset a
    allow: [7]           # allow-list (hard constraint: wait rather than use another card)
    env: { DEBUG: "1" }  # extra env; CUDA_VISIBLE_DEVICES is managed by gpu-hunter
    log: logs/eval_a.log # default is logs/<name>-<timestamp>.log; "-" streams to the terminal

  - name: quick-check
    cmd: ./check.sh
    gpu-num: 1           # omitted fields inherit from defaults
```

Relative `log`/`cwd` paths are resolved from the directory where `gpu-hunter` is started (not from the YAML file); `~` is expanded. Job `name` values are sanitized before being used in generated log filenames.

#### Scheduling semantics

| Mode | Behavior |
|---|---|
| `greedy` (default) | Every tick scans all pending jobs: **jobs that do not fit are skipped and scanning continues** (big jobs do not block small ones). Multiple jobs may launch in the same tick when resources allow, each taking disjoint GPUs. |
| `sequential` | Only the head of the queue is considered; the next job starts after it finishes (including on failure). |

Failure handling: failures are recorded and the queue continues; a summary table is printed when the workflow ends, and **the exit code is the number of failed jobs** (0 when all succeed). A preempted job is rolled back, its GPU history is invalidated and requeued with exponential backoff; after `max-attempts` it is reported as failed. The `try` column shows the attempt count. Pressing Ctrl-C sets the stop flag: no new jobs are started, running jobs are terminated and a partial summary is printed.

## Single-job options

| Option | Default | Description |
|---|---|---|
| `--gpu-num N` | 1 | How many GPUs must be free at the same time |
| `--mem-per-gpu SIZE` | - | Free memory required per GPU, e.g. `40G`; default requires a fully idle card |
| `--prefer LIST` | - | Soft preference: `2,3` tries 2 first, then 3, falling back to other cards |
| `--allow LIST` | all | Hard allow-list: only pick from the listed GPUs |
| `--interval SEC` / `--stable-rounds K` | 5 / 3 | Polling interval / stability window |
| `--idle-mem SIZE` | 500M | Usage below this counts as idle (also requires utilization <= 5% and no compute processes) |
| `--idle-util PCT` | 5 | Utilization at or below this counts as idle |
| `--grace SEC` / `--claim-stable SEC` | 45 / 8 | Launch confirmation grace period / how long a foreign process must persist to count as preemption |
| `--heartbeat SEC` | 120 | Runtime heartbeat interval |
| `-v, --verbose` | off | Full status log (default output uses compact `x` progress) |
| `--log FILE` / `--cwd DIR` | terminal / current directory | Redirect job output / child working directory |

Difference between `prefer` and `allow`:

```
all GPUs -> allow filter (hard) -> stable-idle check -> prefer ranking (soft) -> take the first N
```

- `--prefer 3,2`: 3 is preferred, then 2; if neither is free, use other cards.
- `--allow 2,3`: only 2 and 3 are allowed; other cards stay untouched even when free.
- Multiple GPUs are taken greedily in preference order: `--prefer 3 --gpu-num 2` with {0,2,3} idle -> {3,0}.

## Reserving GPUs (hold mode)

Sometimes the real job cannot be described as a command: interactive work, a
GUI, or a manual sequence of steps. In that case gpu-hunter can *hold* the
cards for you: it waits for N stably idle GPUs, starts one placeholder process
per card that allocates the requested device memory, and keeps it there until
you release it.

```bash
# hold 2 x 12G in the foreground; Ctrl-C releases
gpu-hunter --reserve --gpu-num 2 --mem-per-gpu 12G

# ... later: release and immediately run anything on the same cards
gpu-hunter --release --run -- bash

# or hold in the background and release it later
gpu-hunter --reserve --gpu-num 2 --mem-per-gpu 12G --detach
gpu-hunter --reserve-status
gpu-hunter --release                     # prints CUDA_VISIBLE_DEVICES=...
```

How it behaves:

- The placeholder is a CUDA driver allocation (`cuMemAlloc`) inside a small
  resident process, so no torch/tensorflow dependency is needed. A held card
  shows a compute process and `--mem-per-gpu` of used memory, which means other
  gpu-hunter instances treat it as busy.
- **Foreground (`--reserve`) is the default**: every tick verifies that each
  placeholder is alive and still holds its memory; a lost placeholder is
  replaced when another suitable card becomes stably idle.
- **Ctrl-C / SIGTERM releases**: placeholders get SIGTERM, gpu-hunter waits for
  NVML to confirm the memory is free, removes the state file and prints the
  hand-off.
- An **abnormal exit** (SSH drop, OOM kill, SIGKILL) does *not* release: the
  placeholders run in their own sessions and the state file survives, so
  `--reserve-status` still shows the reservation and `--release` recovers it.
  Every reservation self-expires (`--hold-max`, default 12h; `0` means never).
- `--reserve --run -- CMD` combines both: hold until Ctrl-C, then release and
  immediately run CMD on the same cards (atomic hand-off, no race window).
- `--reserve-full` holds all remaining free memory instead of exactly
  `--mem-per-gpu` -- stronger isolation, but it wastes the rest of the card.
- With `--detach` the command exits after acquiring; supervision is then
  limited to `--reserve-status` plus the self-expiry.
- Reservation state lives in `$XDG_STATE_HOME/gpu-hunter/reservation-<uid>.json`
  (mode 0600). `--release --force` clears stale or unreadable state (after a
  reboot, a hostname change, or if a placeholder pid was reused) and never
  signals a process it cannot verify.

Notes:

- A reservation holds memory and a compute process; it is **not** an exclusive
  GPU lock. A job that still fits in the remaining memory can be started by
  someone who ignores GPU processes.
- Container/MIG setups where the driver API cannot allocate: pass `--hold-cmd`
  with your own placeholder command; `{mem_mb}`, `{gpu}` and `{timeout}` are
  substituted. The same hook is what the test-suite uses to simulate holders.
- A plain `--release` prints `export CUDA_VISIBLE_DEVICES=...`, but the window
  between releasing and your command actually allocating has a small race
  window; prefer `--release --run` for a real hand-off.
- A placeholder allocates the exact `--mem-per-gpu` amount *on top of* the
  CUDA context overhead (a few hundred MB). Asking for nearly all of a card's
  free memory can therefore fail to allocate and retry forever; use a
  slightly smaller value or `--reserve-full` (which keeps ~1G free) instead.

Reserve options: `--reserve`, `--detach`, `--hold-max SEC`, `--reserve-full`,
`--hold-cmd CMD`, `--reserve-status`, `--release`, `--run`, `--force`.

## How it works

1. **Probing**: uses NVML (the library behind `nvidia-smi`) and never parses text output. A card whose process list cannot be read is treated as busy, not idle.
2. **Stability window**: acts only after K consecutive idle polls in a sliding window; multi-GPU requests are filled atomically. A transient NVML error does not reset the window.
3. **Launch**: injects `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` and starts the child in an independent session; inside a workflow, a resource ledger keeps concurrent jobs on disjoint GPUs.
4. **Claim verification**: checks compute processes on the target GPUs during the grace period. Ownership is decided by session id, so bash wrappers, `torchrun` workers and other descendants of the launched child count as ours; a process that is verifiably another session and persists for `claim-stable` seconds triggers a rollback and requeue. Pids that cannot be attributed (containers/namespaces) downgrade to a warning and continue, and are never a reason to kill our job. Early child exit ends verification immediately.
5. **Safety gate**: every signal is sent only after `/proc/<pid>` confirms that the target (and every member of its process group) belongs to the current user and that the pid still refers to the process we launched (pid-reuse check). If ownership or identity cannot be verified, the signal is refused. SIGINT/SIGTERM are forwarded to every child process group; two presses within 5 seconds escalate to SIGKILL; Ctrl-C also stops scheduling new workflow jobs.
6. **Single-instance lock**: flock ensures only one hunter runs per machine.
7. **Reservation**: hold mode records placeholder pids and kernel start times in a small state file and reuses the same ownership/pid-reuse safety gate for every later signal. Placeholders are started in their own sessions and are not tied to the hunter's signal bridge, so an abnormal hunter exit leaves the reservation recoverable instead of silently leaking or losing the cards.

Note: the exit of the direct child marks a job as finished. Do not background the real workload inside `cmd` (for example `bash -c 'python train.py &'`); start it in the foreground so GPU accounting and signals stay correct.

## Running in the background

```bash
nohup gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py &
nohup gpu-hunter workflow.yaml > wf.out 2>&1 &   # job output goes to the per-job log files
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # no real GPU required
```
