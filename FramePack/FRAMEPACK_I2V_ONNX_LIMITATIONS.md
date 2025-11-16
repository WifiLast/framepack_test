# FramePackI2V ONNX Export Limitations

## Current Status: ❌ NOT SUPPORTED

The FramePackI2V HunyuanVideo transformer model **cannot be exported to ONNX** with the current implementation due to fundamental architectural incompatibilities.

## Why ONNX Export Fails

### 1. Dynamic Operations in Rotary Position Embeddings
- **Issue**: Uses `torch.meshgrid` with dynamic tensor inputs
- **Location**: `diffusers_helper/models/hunyuan_video_packed.py:490`
- **Error**: `torch.meshgrid: Expected 0D or 1D tensor in the tensor list`
- **Impact**: ONNX cannot handle dynamic meshgrid operations

### 2. Custom Triton Kernels
- **Issue**: Model uses Triton-compiled quantization kernels
- **Error**: `arange's arguments must be of type tl.constexpr`
- **Impact**: Triton kernels don't export to ONNX (they're Python functions, not traced ops)

### 3. Advanced Attention Mechanisms
- **Issue**: Flash Attention, Sage Attention, xFormers optimizations
- **Impact**: These are custom CUDA kernels that bypass ONNX's operation set

### 4. Model Size & Complexity
- **Model size**: ~24GB (split into 3 files)
- **Parameters**: Billions of parameters
- **Impact**: Even if export worked, the ONNX file would be impractically large

## What Works Instead

### ✅ Recommended Approach 1: PyTorch with torch.compile()

Use PyTorch 2.0+'s `torch.compile()` for TensorRT-like acceleration:

```python
import torch
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# Load model
model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    torch_dtype=torch.bfloat16
).cuda()

# Compile with TorchInductor (TensorRT backend)
model = torch.compile(model, mode='max-autotune', backend='inductor')

# Now use normally - first run will compile, subsequent runs are fast
output = model(hidden_states, timestep, encoder_hidden_states, ...)
```

**Benefits:**
- Native PyTorch support
- Automatic kernel fusion
- Similar speedups to TensorRT
- No ONNX export needed

### ✅ Recommended Approach 2: Quantization

Apply FP8 or INT8 quantization directly in PyTorch:

```python
from transformers import AutoModelForCausalLM
import torch

# Load and quantize
model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    torch_dtype=torch.float8_e4m3fn,  # FP8
    device_map='auto'
)

# Or use bitsandbytes for INT8
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    quantization_config=quantization_config
)
```

### ✅ Recommended Approach 3: Export Individual Components

Export the auxiliary models which DO work with ONNX:

```bash
# Text encoders
python convert_to_onnx.py \
    --model-path tencent/HunyuanVideo \
    --model-type llama_text \
    --subfolder text_encoder \
    --output-name hunyuan_text_encoder.onnx

# CLIP encoder
python convert_to_onnx.py \
    --model-path tencent/HunyuanVideo \
    --model-type clip_text \
    --subfolder text_encoder_2 \
    --output-name hunyuan_clip_encoder.onnx

# VAE (if available)
python convert_to_onnx.py \
    --model-path path/to/vae \
    --model-type vae \
    --output-name hunyuan_vae.onnx
```

These components can be accelerated with ONNX Runtime or TensorRT, while keeping the main transformer in PyTorch.

## Future Possibilities

ONNX export might become feasible if:

1. **PyTorch improves ONNX export** for dynamic operations like meshgrid
2. **Model architecture changes** to avoid incompatible operations
3. **Simplified variant** is released without Triton kernels
4. **ONNX opset expands** to support more dynamic operations

## Conclusion

**For production use, stick with PyTorch + torch.compile() or quantization.**

The infrastructure in `convert_to_onnx.py` is in place and will automatically handle the split model files, but the ONNX export step itself will fail due to the architectural limitations described above.

## See Also

- [convert_to_onnx.py](convert_to_onnx.py) - The conversion tool (loads model but export fails)
- [ONNX_CONVERSION_EXAMPLES.md](ONNX_CONVERSION_EXAMPLES.md) - Full documentation
- [PyTorch torch.compile() docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
