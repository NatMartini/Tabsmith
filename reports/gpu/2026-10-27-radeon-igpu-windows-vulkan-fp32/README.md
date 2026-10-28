# GPU certification: vulkan on AMD Radeon(TM) Graphics

Made by [`tools/gpu_certify.py`](../../../tools/gpu_certify.py) on 2026-10-27: Windows-11-10.0.26300-SP0, CPU AMD Ryzen 9 9950X3D2 16-Core Processor, tabsmith 0.1.0; GPU AMD Radeon(TM) Graphics (2026.04.17 (AMD proprietary shader compiler)), backend `vulkan`, precision `fp32`. Raw numbers and every command in [`results.json`](results.json).

**Result: certified.**

| Check | Result | Evidence |
|---|---|---|
| Determinism: two processes, same weights | pass | model SHA-256 `6a9f0b00acda0150…` twice |
| Exact resume after a crash at step 58 | pass | resumed and uninterrupted weights identical |
| Agreement with the CPU (validation) | pass | balanced_accuracy Δ 0.00e+00 (≤ 0.005), roc_auc Δ 0.00e+00 (≤ 0.002), log_loss Δ 4.90e-09 (≤ 0.002) |
| Bundle + CPU runtime self-check | pass | max \|Δp\| 0.0 |

## Training time

| Problem | CPU train | GPU train | CPU ms/step | GPU ms/step | GPU start-up |
|---|---|---|---|---|---|
| `tabsmith init --sample-data` problem, small-mlp hidden [32, 16], batch 64 | 0.11 s | 1.01 s | 0.1481 | 1.3386 (incl. start-up) | – |
| 100,000 rows (small-mlp, hidden [512, 512], batch 2048, 4 epochs, 136 steps) | 5.14 s | 6.81 s | 35.316 | 44.59 (0.8× faster) | 0.75 s |

The CPU path is single-threaded Flex. On the large problem, ms/step is the marginal
cost of a step: the difference between a 4-epoch and a 1-epoch run, divided by the
difference in steps, so the GPU's one-off start-up (kernel compilation in the first
steps) is reported separately. On the README's problem a step is microseconds of
arithmetic, less than one GPU launch and copy, so the CPU wins and `auto-certified`
keeps it there; the GPU pays off once the work per step grows.

Registry entry (`crates/tabsmith-workflow/src/certified_devices.json`):

```json
{
  "backend": "vulkan",
  "device": "AMD Radeon(TM) Graphics",
  "os": "windows",
  "driver": "2026.04.17 (AMD proprietary shader compiler)",
  "precision": "fp32",
  "certified_on": "2026-10-27",
  "evidence": "reports/gpu/2026-10-27-radeon-igpu-windows-vulkan-fp32"
}
```
