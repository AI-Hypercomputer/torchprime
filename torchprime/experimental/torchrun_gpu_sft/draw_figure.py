"""Visualize GPU and TPU training metrics.

This script combines metrics logged by the TPU training job (``data_tp.txt``)
and a GPU run (``data_hf.txt``) into a single matplotlib figure. The GPU file is
expected to contain one Python ``dict`` per line. For that file the x-axis is
the line number, whereas the TPU file uses the ``step`` index extracted from the
log line.

Run this script in the same directory as the two data files:

```
python draw_figure.py
```
"""

import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------- locate files ----------
script_dir = Path(__file__).resolve().parent
tp_path = script_dir / "data_tp.txt"
hf_path = script_dir / "data_hf.txt"

# Ensure the required log files are present.
for path in (tp_path, hf_path):
  if not path.exists():
    sys.exit(f"File not found: {path}")

# ---------- parse TorchPrime step log ----------
step_re = re.compile(
  r"step:\s*(\d+),\s*loss:\s*([\d.]+),\s*grad_norm:\s*([\d.]+),"
  r"\s*lr:\s*([\d.eE+-]+)"
)
tp_step, tp_loss, tp_grad, tp_lr = [], [], [], []
with tp_path.open("r", encoding="utf-8") as fh:
  for line in fh:
    m = step_re.search(line)
    if not m:
      continue
    s, ls, g, lr_val = m.groups()
    tp_step.append(int(s) + 1)
    tp_loss.append(float(ls))
    tp_grad.append(float(g))
    tp_lr.append(float(lr_val))

if not tp_step:
  sys.exit("data_tp.txt: no matching log lines found")

# ---------- parse HF dict-per-line log (x-axis = row number) ----------
hf_step, hf_loss, hf_grad, hf_lr = [], [], [], []
with hf_path.open("r", encoding="utf-8") as fh:
  for idx, raw in enumerate(fh, start=0):  # row index = step
    raw = raw.strip()
    if not raw:
      continue
    d = ast.literal_eval(raw)
    hf_step.append(idx)
    hf_loss.append(float(d["loss"]))
    hf_grad.append(float(d["grad_norm"]))
    hf_lr.append(float(d["learning_rate"]))

if not hf_step:
  sys.exit("data_hf.txt: no valid dict lines found")

# ---------- plotting ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax_loss, ax_grad, ax_lr, ax_blank = axes.flatten()

ax_loss.plot(tp_step, tp_loss, label="TPU loss", marker="o")
ax_loss.plot(hf_step, hf_loss, label="GPU loss", marker="x")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Step")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend()

ax_grad.plot(tp_step, tp_grad, label="TPU grad_norm", marker="o")
ax_grad.plot(hf_step, hf_grad, label="GPU grad_norm", marker="x")
ax_grad.set_title("Gradient Norm")
ax_grad.set_xlabel("Step")
ax_grad.set_ylim(0, 32)
ax_grad.grid(True)
ax_grad.legend()

ax_lr.plot(tp_step, tp_lr, label="TPU LR", marker="o")
ax_lr.plot(hf_step, hf_lr, label="GPU LR", marker="x")
ax_lr.set_title("Learning Rate")
ax_lr.set_xlabel("Step")
ax_lr.set_ylabel("LR")
ax_lr.grid(True)
ax_lr.legend()

ax_blank.axis("off")  # unused subplot

fig.tight_layout()
out_path = script_dir / "figure_combined.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved combined plot → {out_path}")
