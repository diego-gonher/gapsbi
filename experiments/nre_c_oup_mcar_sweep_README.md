# OUP NRE-C MCAR-10% sweep

`nre_c_oup_mcar_sweep.yaml` is the experiment contract: five NRE-C methods, three budgets, five seeds, and reference IDs 0–9. The launcher creates a single-seed config for each matrix row from that method's low-budget template, changing only dataset/output paths, budget limits, batch size from the matrix, seed, and reference selection.

From the repository root:

```bash
# Show all 75 jobs and what each would run, without starting work.
./experiments/nre_c_oup_mcar_sweep.sh --dry-run

# Launch/resume sequentially in the foreground.
./experiments/nre_c_oup_mcar_sweep.sh

# Inspect progress or failures.
./experiments/nre_c_oup_mcar_sweep.sh --progress
./experiments/nre_c_oup_mcar_sweep.sh --failures
```

The launcher takes an exclusive PID lock, runs one training/evaluation process at a time, and writes per-run logs, generated configs, progress, and failure records under `logs/nre_c_oup_mcar_10pct/`. A second invocation is safe: validator-confirmed complete runs are skipped; a matching checkpoint with incomplete evaluation is reused and evaluated for all ten references. A two-reference smoke result is incomplete. Corrupt or mismatched artifacts are recorded as failures and left untouched. If interrupted during training, a later invocation trains only if no checkpoint was saved; if interrupted after checkpoint creation, it resumes at evaluation.

To keep a local Mac awake during an overnight run:

```bash
mkdir -p logs/nre_c_oup_mcar_10pct
nohup caffeinate -dims ./experiments/nre_c_oup_mcar_sweep.sh \
  > logs/nre_c_oup_mcar_10pct/sweep.log 2>&1 < /dev/null &
echo $!
```

The process ID is the `caffeinate` wrapper; verify it with `ps -p <PID> -o pid=,etime=,command=` and inspect the progress command and `logs/nre_c_oup_mcar_10pct/sweep.log`. Re-running the launch command while the lock is held exits without creating a duplicate campaign process.
