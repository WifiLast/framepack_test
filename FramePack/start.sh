conda activate py310
FRAMEPACK_FAST_START=1 FRAMEPACK_PRELOAD_REPOS=0   python demo_gradio.py --fast-start --disable-fbcache --disable-sim-cache --disable-kv-cache --chunk-transfer-limit-mb 256 --use-memory-v2


#FRAMEPACK_FAST_START=1 FRAMEPACK_ENABLE_OPT_CACHE=1 FRAMEPACK_ENABLE_COMPILE=1 FRAMEPACK_ENABLE_FBCACHE=1 FRAMEPACK_ENABLE_SIM_CACHE=1 FRAMEPACK_ENABLE_KV_CACHE=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 python demo_gradio.py --fast-start --xformers-mode aggressive

# Optimierte Version ohne JIT (funktioniert besser mit begrenztem RAM):
#FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 FRAMEPACK_ENABLE_FBCACHE=1 FRAMEPACK_ENABLE_SIM_CACHE=1 FRAMEPACK_ENABLE_KV_CACHE=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 python demo_gradio.py --fast-start --xformers-mode aggressive

# Extreme Memory Saving Mode (für 110GB+ Modelle mit 16GB VRAM):
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 FRAMEPACK_BNB_CPU_OFFLOAD=1 python demo_gradio.py --fast-start --xformers-mode standard