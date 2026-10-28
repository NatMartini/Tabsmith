# GPU certifications / GPU 认证

Each directory is one run of [`tools/gpu_certify.py`](../../tools/gpu_certify.py): the
real `tabsmith` binary, built with `--features gpu`, trains on the CPU and on one GPU
path, and four checks decide whether the path is certified:
1. two processes train bit-identical weights;
2. a run killed mid-training and resumed in a new process equals an uninterrupted run;
3. the validation metrics agree with the CPU's;
4. the GPU-trained bundle passes the CPU runtime's self-check.

A certified path gets its entry in
[`certified_devices.json`](../../crates/tabsmith-workflow/src/certified_devices.json).
That entry is what lets `execution.candidate_device = "auto-certified"` use the path.

| Path | Result | ms/step, 512×512 network, batch 2048 | vs the CPU engine |
|---|---|---|---|
| [RTX 5090, Vulkan, fp32](2026-10-27-rtx5090-windows-vulkan-fp32/) | certified | 4.8 | 6.6× faster |
| [RTX 5090, CUDA, TF32](2026-10-27-rtx5090-windows-cuda-tf32/) | certified | 2.9 | 10.8× faster |
| [AMD Radeon integrated, Vulkan, fp32](2026-10-27-radeon-igpu-windows-vulkan-fp32/) | certified | 44.6 | 0.8× (slower; `auto-certified` never picks it) |

Machine: Windows 11 Pro 10.0.26300, AMD Ryzen 9 9950X3D2 (the CPU reference is
single-threaded, 32–35 ms per step). NVIDIA driver 610.88, CUDA 12.8. The code is the
working tree of the commit that adds these reports.

Certified means *these* paths on *this* hardware, OS and driver. Another driver version
of the same GPU is reported as not certified, and the message names the certified
version.

中文：每个目录是 `tools/gpu_certify.py` 的一次运行。用启用了 `--features gpu` 的真实 `tabsmith` 二进制在 CPU
和一条 GPU 路径上训练，并检查四项：两个进程训练出逐位相同的权重；中途被杀死、在新进程中续训的结果与
不中断训练相同；验证指标与 CPU 一致；GPU 训练的模型包通过 CPU 运行时自检。通过的路径写入
`certified_devices.json`，`auto-certified` 才会使用它。上表三条路径均已认证：RTX 5090 的 CUDA（TF32）每步比
CPU 快 10.8 倍，Vulkan（fp32）快 6.6 倍；AMD 核显比 CPU 慢，因此 `auto-certified` 永远不会选择它。
