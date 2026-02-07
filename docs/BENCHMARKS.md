# Benchmark Results

## VSA Pipeline Performance (Validated v2.1.0)

| Metric               | Value |
|----------------------|-------|
| Baseline Avg F1      | 82.4% |
| Baseline Avg Recall  | 97.1% |
| Resonator Convergence| 100%  |
| Expected GPU Speedup | 5-10x (VSA Resonator on NVIDIA T4) |

## Hardware

- Instance type: AWS g4dn.xlarge (NVIDIA T4 GPU)
- CUDA: 12.1
- TORCH_CUDA_ARCH_LIST: 7.5

## Pipeline Throughput

- 156K+ rows processed in single-pass bundling with pre-normalized alias lookup
- Target: < 15s end-to-end on 4 vCPU / 16 GB Fargate task
