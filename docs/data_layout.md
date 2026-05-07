# Data and Checkpoint Layout

Large datasets, checkpoints, and generated evaluation reports are not tracked by Git.

Recommended AutoDL layout:

```text
~/minimind/dataset/                 # local training/evaluation jsonl files
~/minimind/dataset/examples/        # tiny reproducible examples tracked by Git
~/minimind/checkpoints/             # training checkpoints, ignored by Git
~/minimind/out/                     # exported weights, ignored by Git
~/minimind/evals/reports/           # generated evaluation reports, ignored by Git
~/minimind/evals/reports0/          # old generated evaluation reports, ignored by Git
> EOF

