conda activate py310
# Empfohlene Konfiguration (ohne TensorRT):
FRAMEPACK_FAST_START=1 FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 FRAMEPACK_BNB_CPU_OFFLOAD=1 FRAMEPACK_VAE_CHUNK_SIZE=2 python demo_gradio.py --fast-start --xformers-mode standard --use-memory-v2

#FRAMEPACK_FAST_START=1 FRAMEPACK_ENABLE_OPT_CACHE=1 FRAMEPACK_ENABLE_COMPILE=1 FRAMEPACK_ENABLE_FBCACHE=1 FRAMEPACK_ENABLE_SIM_CACHE=1 FRAMEPACK_ENABLE_KV_CACHE=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 python demo_gradio.py --fast-start --xformers-mode aggressive

# Optimierte Version ohne JIT (funktioniert besser mit begrenztem RAM):
#FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 FRAMEPACK_ENABLE_FBCACHE=1 FRAMEPACK_ENABLE_SIM_CACHE=1 FRAMEPACK_ENABLE_KV_CACHE=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 python demo_gradio.py --fast-start --xformers-mode aggressive

# Extreme Memory Saving Mode (für 110GB+ Modelle mit 16GB VRAM):
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=128 FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 FRAMEPACK_BNB_CPU_OFFLOAD=1 python demo_gradio.py --fast-start --xformers-mode standard

# Standard mit BNB 4-bit + Memory V2:
#FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 FRAMEPACK_BNB_CPU_OFFLOAD=1 FRAMEPACK_VAE_CHUNK_SIZE=2 python demo_gradio.py --fast-start --xformers-mode standard  --use-memory-v2

FRAMEPACK_USE_BNB=1                      # Enable BitsAndBytes (incompatible with TensorRT)
FRAMEPACK_BNB_LOAD_IN_4BIT=1             # 4-bit quantization (incompatible with TensorRT)
FRAMEPACK_BNB_CPU_OFFLOAD=1              # CPU offload (incompatible with TensorRT)

PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 PYTORCH_ENABLE_FLASH_SDP=0 FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=0 FRAMEPACK_USE_BNB=0 FRAMEPACK_BNB_LOAD_IN_4BIT=0 FRAMEPACK_BNB_CPU_OFFLOAD=1 FRAMEPACK_VAE_CHUNK_SIZE=2 python demo_gradio.py --fast-start --xformers-mode aggressive --use-memory-v2

# Mit TensorRT (Flash Attention deaktivieren, um CUDA-Fehler zu vermeiden):
#FRAMEPACK_TRT_WORKSPACE_MB=1024 PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 PYTORCH_ENABLE_FLASH_SDP=0 FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=0 FRAMEPACK_USE_BNB=0 FRAMEPACK_BNB_LOAD_IN_4BIT=0 FRAMEPACK_BNB_CPU_OFFLOAD=1 FRAMEPACK_VAE_CHUNK_SIZE=2 python demo_gradio.py --fast-start --xformers-mode aggressive --use-memory-v2 --enable-tensorrt --tensorrt-transformer

# --tensorrt-text-encoders

# FRAMEPACK_ENABLE_BETTERTRANSFORMER=1
--enable-profiling  --profiling-iterations 5 --profiling-output-dir ./my_profile

python demo_gradio.py --fast-start --enable-tensorrt --tensorrt-transformer --use-memory-v2