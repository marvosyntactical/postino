# Source before working: `source env.sh`. Keeps large files off the 100 GB home quota.
export HF_HOME=/export/scratch/mkoss/hf
export UV_CACHE_DIR=/export/scratch/mkoss/uv-cache
export PI_RUNS_DIR=/export/data/mkoss/postino/runs
export TOKENIZERS_PARALLELISM=false
# vLLM must not fork after torch has started OpenMP threads (segfaults in gomp_team_start).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# FlashInfer's sampler JIT-compiles with nvcc; the system nvcc (12.2) is too old and the venv's CUDA 13
# wheels mix header versions. Use vLLM's PyTorch sampler instead (attention on A100 uses precompiled FlashAttention).
export VLLM_USE_FLASHINFER_SAMPLER=0
