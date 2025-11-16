#!/usr/bin/env python3
"""
Command-line tool for converting PyTorch models to ONNX format.

Usage:
    python convert_to_onnx.py --model-path <path> --model-type <type> [options]

Example:
    python convert_to_onnx.py \
        --model-path hf_download/hub/models--lllyasviel--flux_redux_bfl/ \
        --model-type flux_redux \
        --output-dir Cache/onnx_models/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from diffusers_helper.onnx_converter import (
    export_model_to_onnx,
    get_onnx_cache_dir,
)

DEFAULT_SD_V1_CONFIG = """model:
  base_learning_rate: 1.0e-04
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 10000 ]
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
"""


def load_flux_redux_model(model_path: str) -> Tuple[nn.Module, dict]:
    """
    Load Flux Redux model from the specified path.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, config_dict)
    """
    from transformers import AutoModel, AutoConfig

    print(f"Loading Flux Redux model from: {model_path}")

    try:
        # Try loading as transformers model
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()

        config_dict = {
            'hidden_size': getattr(config, 'hidden_size', 768),
            'num_layers': getattr(config, 'num_hidden_layers', 12),
        }

        print(f"  Model type: {type(model).__name__}")
        print(f"  Config: {config_dict}")

        return model, config_dict

    except Exception as e:
        print(f"Failed to load as transformers model: {e}")
        print("Attempting to load as raw PyTorch model...")

        # Try loading as raw PyTorch checkpoint
        model_path_obj = Path(model_path)
        checkpoint_files = list(model_path_obj.glob("*.pt")) + list(model_path_obj.glob("*.pth"))

        if checkpoint_files:
            print(f"  Found checkpoint file: {checkpoint_files[0]}")
            model = torch.load(checkpoint_files[0], map_location='cpu')
            return model, {}

        # Check if this is a HuggingFace cache directory with snapshots
        snapshots_dir = model_path_obj / "snapshots"
        print(f"\nDEBUG: Checking for snapshots directory: {snapshots_dir}")
        print(f"  snapshots_dir.exists() = {snapshots_dir.exists()}")

        if snapshots_dir.exists():
            print(f"  Found snapshots directory, checking for model files...")
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            print(f"  Found {len(snapshot_dirs)} snapshot directories")

            if snapshot_dirs:
                # Use the first (and usually only) snapshot
                snapshot_path = snapshot_dirs[0]
                print(f"  Trying snapshot: {snapshot_path}")

                # List contents of snapshot
                print(f"  Snapshot contents:")
                for item in snapshot_path.iterdir():
                    print(f"    - {item.name}")

                try:
                    config = AutoConfig.from_pretrained(str(snapshot_path), trust_remote_code=True)
                    model = AutoModel.from_pretrained(
                        str(snapshot_path),
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
                    model.eval()
                    config_dict = {
                        'hidden_size': getattr(config, 'hidden_size', 768),
                        'num_layers': getattr(config, 'num_hidden_layers', 12),
                    }
                    return model, config_dict
                except Exception as snapshot_e:
                    print(f"  Failed to load from snapshot: {snapshot_e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  WARNING: snapshots directory is empty or contains no subdirectories")

        # List what's actually in the directory for debugging
        print(f"\nDEBUG: Contents of {model_path}:")
        try:
            for item in model_path_obj.iterdir():
                print(f"  - {item.name}")
        except Exception:
            pass

        raise RuntimeError(f"Could not load model from {model_path}\n"
                         f"Expected either:\n"
                         f"  - config.json + model files (*.safetensors, *.bin)\n"
                         f"  - PyTorch checkpoint (*.pt, *.pth)\n"
                         f"  - HuggingFace cache with snapshots/ subdirectory")


def load_diffusers_model(model_path: str, subfolder: Optional[str] = None) -> Tuple[nn.Module, dict]:
    """
    Load a diffusers model (VAE, UNet, etc.).

    Args:
        model_path: Path to the model directory
        subfolder: Optional subfolder within model_path

    Returns:
        Tuple of (model, config_dict)
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel

    print(f"Loading diffusers model from: {model_path}")
    if subfolder:
        print(f"  Subfolder: {subfolder}")

    # Try VAE
    try:
        model = AutoencoderKL.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype=torch.float16,
        )
        model.eval()
        config_dict = {
            'in_channels': model.config.in_channels,
            'latent_channels': model.config.latent_channels,
        }
        print(f"  Loaded as AutoencoderKL")
        return model, config_dict
    except:
        pass

    # Try UNet
    try:
        model = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype=torch.float16,
        )
        model.eval()
        config_dict = {
            'in_channels': model.config.in_channels,
            'out_channels': model.config.out_channels,
        }
        print(f"  Loaded as UNet2DConditionModel")
        return model, config_dict
    except:
        pass

    raise RuntimeError(f"Could not load diffusers model from {model_path}")


def ensure_default_sd_config(checkpoint_path: str) -> Path:
    """
    Ensure a default v1 Stable Diffusion config YAML exists on disk.

    Args:
        checkpoint_path: Path to the checkpoint (used for naming)

    Returns:
        Path to the generated YAML file
    """
    cache_dir = get_onnx_cache_dir() / "auto_sd_configs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = cache_dir / f"{Path(checkpoint_path).stem}_auto_v1-inference.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(DEFAULT_SD_V1_CONFIG)
        print(f"  Generated default Stable Diffusion v1 config at: {cfg_path}")
    else:
        print(f"  Using cached auto-generated config at: {cfg_path}")

    return cfg_path


def load_sd_checkpoint_component(
    checkpoint_path: str,
    component: str,
    sd_config_path: Optional[str] = None,
) -> Tuple[nn.Module, dict]:
    """
    Load a Stable Diffusion component directly from a single-file checkpoint.

    Args:
        checkpoint_path: Path to .safetensors/.ckpt file
        component: Component name ('unet', 'vae', 'vae_encoder', 'vae_decoder', 'clip_text')
        sd_config_path: Optional original config YAML (required for SD1.x checkpoints)

    Returns:
        Tuple of (model, config_dict)
    """
    from diffusers import StableDiffusionPipeline

    print(f"Loading {component} from Stable Diffusion checkpoint: {checkpoint_path}")
    if sd_config_path:
        print(f"  Using original config: {sd_config_path}")

    pipe_kwargs = {'torch_dtype': torch.float16}
    if sd_config_path:
        pipe_kwargs['original_config_file'] = sd_config_path

    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint_path,
        **pipe_kwargs,
    )

    component = component.lower()
    if component in ['vae', 'vae_encoder', 'vae_decoder']:
        model = pipe.vae
        config_dict = {
            'in_channels': model.config.in_channels,
            'latent_channels': model.config.latent_channels,
        }
    elif component == 'unet':
        model = pipe.unet
        config_dict = {
            'in_channels': model.config.in_channels,
            'out_channels': model.config.out_channels,
        }
    elif component == 'clip_text':
        if not hasattr(pipe, 'text_encoder') or pipe.text_encoder is None:
            raise RuntimeError("Checkpoint does not contain a CLIP text encoder")
        model = pipe.text_encoder
        config_dict = {
            'hidden_size': model.config.hidden_size,
            'vocab_size': model.config.vocab_size,
        }
    else:
        raise ValueError(f"Unsupported component '{component}' for checkpoint loading")

    model.eval()

    # Free unnecessary parts of the pipeline
    del pipe

    return model, config_dict


def load_text_encoder(model_path: str, encoder_type: str = 'clip') -> Tuple[nn.Module, dict]:
    """
    Load a text encoder model.

    Args:
        model_path: Path to the model directory
        encoder_type: Type of encoder ('clip', 'llama', 't5')

    Returns:
        Tuple of (model, config_dict)
    """
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoModel,
    )

    print(f"Loading {encoder_type} text encoder from: {model_path}")

    if encoder_type.lower() == 'clip':
        try:
            model = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
        except:
            model = CLIPTextModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
    else:
        # Generic auto loader
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    model.eval()

    config_dict = {
        'hidden_size': model.config.hidden_size,
        'vocab_size': model.config.vocab_size,
    }

    print(f"  Model type: {type(model).__name__}")
    return model, config_dict


def load_framepack_i2v_model(model_path: str) -> Tuple[nn.Module, dict]:
    """
    Load FramePackI2V HunyuanVideo transformer model.
    This model is split into 3 safetensors files that are automatically loaded by from_pretrained.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, config_dict)
    """
    # Import the custom model class
    sys.path.insert(0, str(Path(__file__).parent))
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

    print(f"Loading FramePackI2V HunyuanVideo model from: {model_path}")
    print("  Note: This model is split into 3 safetensors files which will be loaded automatically")
    print("  WARNING: ONNX export of this large transformer model has known issues:")
    print("    - torch.meshgrid in rotary embeddings is not fully ONNX-compatible")
    print("    - The model is very large (~24GB) and complex")
    print("    - Consider exporting individual components (text encoders, VAE) instead")
    print("    - For TensorRT acceleration, use PyTorch directly with torch.compile")

    try:
        model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        model.eval()

        config_dict = {
            'in_channels': model.config.in_channels if hasattr(model.config, 'in_channels') else 16,
            'out_channels': model.config.out_channels if hasattr(model.config, 'out_channels') else 16,
        }

        print(f"  Model type: {type(model).__name__}")
        print(f"  Config: {config_dict}")

        return model, config_dict

    except Exception as e:
        print(f"Failed to load FramePackI2V model: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Could not load FramePackI2V model from {model_path}\n"
                         f"Make sure the model files are present and the path is correct.")


def get_sample_inputs_flux_redux(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """Generate sample inputs for Flux Redux model (SiglipVisionModel expects pixel values)."""
    batch_size = 1

    # SiglipVisionModel expects pixel_values input (images)
    # Default Siglip image size is 384x384 with 3 channels
    image_size = 384
    num_channels = 3

    pixel_values = torch.randn(
        batch_size, num_channels, image_size, image_size,
        dtype=torch.float16,
        device=device
    )

    return (pixel_values,)


def get_sample_inputs_text_encoder(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """Generate sample inputs for text encoder."""
    batch_size = 1
    seq_length = 77  # Standard CLIP length
    vocab_size = config.get('vocab_size', 49408)

    input_ids = torch.randint(
        0, vocab_size,
        (batch_size, seq_length),
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.ones(
        batch_size, seq_length,
        dtype=torch.long,
        device=device
    )

    return (input_ids, attention_mask)


def get_sample_inputs_vae_encoder(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """Generate sample inputs for VAE encoder."""
    batch_size = 1
    in_channels = config.get('in_channels', 3)
    height, width = 512, 512

    sample = torch.randn(
        batch_size, in_channels, height, width,
        dtype=torch.float16,
        device=device
    )

    return (sample,)


def get_sample_inputs_vae_decoder(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """Generate sample inputs for VAE decoder."""
    batch_size = 1
    latent_channels = config.get('latent_channels', 4)
    height, width = 64, 64  # Latent space dimensions

    latents = torch.randn(
        batch_size, latent_channels, height, width,
        dtype=torch.float16,
        device=device
    )

    return (latents,)


def get_sample_inputs_unet(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """Generate sample inputs for UNet diffusion model."""
    batch_size = 1
    in_channels = getattr(model.config, 'in_channels', config.get('in_channels', 4))
    spatial = getattr(model.config, 'sample_size', 64)
    if isinstance(spatial, (list, tuple)):
        height, width = spatial
    else:
        height = width = spatial

    sample = torch.randn(
        batch_size, in_channels, height, width,
        dtype=torch.float16,
        device=device,
    )

    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)

    seq_length = 77
    cross_attention_dim = getattr(model.config, 'cross_attention_dim', config.get('cross_attention_dim', 768))
    encoder_hidden_states = torch.randn(
        batch_size, seq_length, cross_attention_dim,
        dtype=torch.float16,
        device=device,
    )

    return (sample, timestep, encoder_hidden_states)


class VAEEncoderWrapper(nn.Module):
    """Wrapper to expose only the encoder portion of AutoencoderKL."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        latent_dist = self.vae.encode(sample).latent_dist
        return latent_dist.mean


class VAEDecoderWrapper(nn.Module):
    """Wrapper to expose only the decoder portion of AutoencoderKL."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.scaling_factor = getattr(vae, 'scaling_factor', getattr(vae.config, 'scaling_factor', 0.18215))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        scaled_latents = latents / self.scaling_factor
        decoded = self.vae.decode(scaled_latents, return_dict=False)[0]
        return decoded


def get_sample_inputs_framepack_i2v(model: nn.Module, config: dict, device: str = 'cuda') -> Tuple[Any, ...]:
    """
    Generate sample inputs for FramePackI2V HunyuanVideo transformer.

    Forward signature:
        forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                pooled_projections, guidance, latent_indices=None, clean_latents=None,
                clean_latent_indices=None, clean_latents_2x=None, clean_latent_2x_indices=None,
                clean_latents_4x=None, clean_latent_4x_indices=None, image_embeddings=None,
                attention_kwargs=None, return_dict=True)
    """
    import os

    # Note: For ONNX export, we need to disable certain features that use dynamic operations
    # like torch.meshgrid in rotary embeddings and Triton kernels

    # Disable Triton compilation which is incompatible with ONNX
    os.environ['TRITON_DISABLE'] = '1'

    # Set attention mode to standard (disable flash/sage attention)
    try:
        from diffusers_helper.models.hunyuan_video_packed import set_attention_accel_mode
        set_attention_accel_mode('standard')
        print("  Set attention acceleration mode to 'standard' for ONNX compatibility")
    except:
        pass

    # Temporarily disable model features that are incompatible with ONNX
    original_image_proj = getattr(model, 'image_projection', None)

    # Disable image projection to avoid meshgrid issues
    if hasattr(model, 'image_projection'):
        model.image_projection = None
        print("  Disabled image_projection to avoid meshgrid issues")

    batch_size = 1
    in_channels = config.get('in_channels', 16)
    # Use patch_size_t = 1 for simpler export (typical value)
    num_frames = 1
    height, width = 16, 32  # Smaller size for ONNX export to reduce memory

    # Main latent input
    hidden_states = torch.randn(
        batch_size, in_channels, num_frames, height, width,
        dtype=torch.bfloat16,
        device=device
    )

    # Timestep - use scalar instead of tensor
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=device)

    # Text encoder hidden states
    text_seq_length = 128  # Reduced for ONNX export
    text_hidden_size = 4096  # Text embeddings dimension
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_length, text_hidden_size,
        dtype=torch.bfloat16,
        device=device
    )

    # Attention mask for text
    encoder_attention_mask = torch.ones(
        batch_size, text_seq_length,
        dtype=torch.bool,
        device=device
    )

    # Pooled text projections - THIS IS 768, NOT 4096!
    # See: pooled_projection_dim in HunyuanVideoTransformer3DModelPacked.__init__
    pooled_projection_dim = 768
    pooled_projections = torch.randn(
        batch_size, pooled_projection_dim,
        dtype=torch.bfloat16,
        device=device
    )

    # Guidance scale
    guidance = torch.tensor([6.0], dtype=torch.bfloat16, device=device)

    return (
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    )


def detect_sd_model_version(model_path: str) -> str:
    """
    Detect Stable Diffusion model version from checkpoint.

    Args:
        model_path: Path to the .safetensors/.ckpt file

    Returns:
        Version string: 'sd1', 'sd2', 'sdxl', or 'unknown'
    """
    try:
        from safetensors import safe_open

        path_obj = Path(model_path)
        if not path_obj.exists():
            return 'unknown'

        # Load state dict keys to detect architecture
        if path_obj.suffix.lower() == '.safetensors':
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
        else:
            # For .ckpt files
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                keys = list(checkpoint['state_dict'].keys())
            else:
                keys = list(checkpoint.keys())

        # Detect based on key patterns
        # SDXL has more blocks and different structure
        if any('conditioner' in k for k in keys):
            return 'sdxl'
        elif any('model.diffusion_model.input_blocks.8' in k for k in keys):
            # SD2.x has deeper UNet
            return 'sd2'
        elif any('model.diffusion_model.input_blocks.0.0.weight' in k for k in keys):
            # SD1.x
            return 'sd1'
        else:
            return 'unknown'

    except Exception as e:
        print(f"Warning: Could not detect model version: {e}")
        return 'unknown'


def convert_model_to_onnx(
    model_path: str,
    model_type: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    device: str = 'cuda',
    opset_version: int = 17,
    subfolder: Optional[str] = None,
    sd_config_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Convert a model to ONNX format.

    Args:
        model_path: Path to the model directory
        model_type: Type of model (flux_redux, clip_text, vae_encoder, vae_decoder, etc.)
        output_dir: Output directory for ONNX file (default: uses cache dir)
        output_name: Custom output filename (default: auto-generated)
        device: Device to use for conversion ('cuda' or 'cpu')
        opset_version: ONNX opset version
        subfolder: Optional subfolder within model_path

    Returns:
        Path to converted ONNX file, or None if conversion failed
    """

    # Load model based on type
    print(f"\n{'='*80}")
    print(f"Converting {model_type} model to ONNX")
    print(f"{'='*80}\n")

    model_loaders = {
        'flux_redux': load_flux_redux_model,
        'clip_text': lambda p: load_text_encoder(p, 'clip'),
        'llama_text': lambda p: load_text_encoder(p, 'llama'),
        't5_text': lambda p: load_text_encoder(p, 't5'),
        'vae': load_diffusers_model,
        'vae_encoder': load_diffusers_model,
        'vae_decoder': load_diffusers_model,
        'unet': load_diffusers_model,
        'framepack_i2v': load_framepack_i2v_model,
    }

    sample_input_generators = {
        'flux_redux': get_sample_inputs_flux_redux,
        'clip_text': get_sample_inputs_text_encoder,
        'llama_text': get_sample_inputs_text_encoder,
        't5_text': get_sample_inputs_text_encoder,
        'vae_encoder': get_sample_inputs_vae_encoder,
        'vae_decoder': get_sample_inputs_vae_decoder,
        'vae': get_sample_inputs_vae_encoder,
        'unet': get_sample_inputs_unet,
        'framepack_i2v': get_sample_inputs_framepack_i2v,
    }

    input_names_map = {
        'flux_redux': ['pixel_values'],
        'clip_text': ['input_ids', 'attention_mask'],
        'llama_text': ['input_ids', 'attention_mask'],
        't5_text': ['input_ids', 'attention_mask'],
        'vae_encoder': ['sample'],
        'vae_decoder': ['latents'],
        'vae': ['sample'],
        'unet': ['sample', 'timestep', 'encoder_hidden_states'],
        'framepack_i2v': [
            'hidden_states',
            'timestep',
            'encoder_hidden_states',
            'encoder_attention_mask',
            'pooled_projections',
            'guidance',
        ],
    }

    output_names_map = {
        'flux_redux': ['last_hidden_state', 'pooler_output'],
        'clip_text': ['last_hidden_state', 'pooler_output'],
        'llama_text': ['last_hidden_state'],
        't5_text': ['last_hidden_state'],
        'vae_encoder': ['latent_dist'],
        'vae_decoder': ['sample'],
        'vae': ['latent_dist'],
        'unet': ['sample'],
        'framepack_i2v': ['output'],
    }

    # Load model
    loader = model_loaders.get(model_type)
    if loader is None:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Supported types: {', '.join(model_loaders.keys())}")
        return None

    model_path_obj = Path(model_path)
    checkpoint_exts = {'.safetensors', '.ckpt', '.bin'}
    is_checkpoint_file = model_path_obj.is_file() and model_path_obj.suffix.lower() in checkpoint_exts
    sd_checkpoint_components = {'unet', 'vae', 'vae_encoder', 'vae_decoder', 'clip_text'}

    try:
        if is_checkpoint_file:
            if model_type not in sd_checkpoint_components:
                print(f"Error: Checkpoint loading is only supported for: {', '.join(sorted(sd_checkpoint_components))}")
                return None
            if sd_config_path is None:
                print("INFO: --sd-config-path not provided. Auto-generating default v1 config...")
                sd_config_path = str(ensure_default_sd_config(model_path))
            model, config = load_sd_checkpoint_component(
                checkpoint_path=model_path,
                component=model_type,
                sd_config_path=sd_config_path,
            )
        elif subfolder and model_type in ['vae', 'vae_encoder', 'vae_decoder', 'unet']:
            model, config = load_diffusers_model(model_path, subfolder)
        else:
            model, config = loader(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Wrap specific VAE components
    if model_type == 'vae_decoder':
        model = VAEDecoderWrapper(model)
    elif model_type == 'vae_encoder':
        model = VAEEncoderWrapper(model)

    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\nMoving model to CUDA...")
        model = model.to('cuda')
    else:
        device = 'cpu'
        print(f"\nUsing CPU for conversion...")

    # Generate sample inputs
    input_generator = sample_input_generators.get(model_type)
    if input_generator is None:
        print(f"Error: No sample input generator for model type '{model_type}'")
        return None

    try:
        print(f"Generating sample inputs...")
        sample_inputs = input_generator(model, config, device)
        print(f"  Input shapes: {[tuple(x.shape) if torch.is_tensor(x) else x for x in sample_inputs]}")
    except Exception as e:
        print(f"Error generating sample inputs: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get input/output names
    input_names = input_names_map.get(model_type, ['input'])
    output_names = output_names_map.get(model_type, ['output'])

    # Dynamic axes for flexible batch/sequence sizes
    dynamic_axes = {}
    if model_type in ['clip_text', 'llama_text', 't5_text']:
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'},
        }
    elif model_type == 'flux_redux':
        dynamic_axes = {
            'pixel_values': {0: 'batch'},
            'last_hidden_state': {0: 'batch'},
            'pooler_output': {0: 'batch'},
        }
    elif model_type == 'framepack_i2v':
        dynamic_axes = {
            'hidden_states': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
            'encoder_hidden_states': {0: 'batch', 1: 'text_sequence'},
            'encoder_attention_mask': {0: 'batch', 1: 'text_sequence'},
            'pooled_projections': {0: 'batch'},
            'output': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
        }

    # Determine output path
    if output_dir is None:
        output_dir = get_onnx_cache_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        onnx_path = output_dir / output_name
        if not onnx_path.suffix:
            onnx_path = onnx_path.with_suffix('.onnx')
    else:
        model_path_obj = Path(model_path)
        model_base_name = model_path_obj.stem if model_path_obj.is_file() else model_path_obj.name
        model_name = f"{model_type}_{model_base_name}"
        onnx_path = output_dir / f"{model_name}.onnx"

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    print(f"  Output: {onnx_path}")

    result = export_model_to_onnx(
        model=model,
        model_name=model_type,
        sample_inputs=sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=opset_version,
    )

    # Clean up
    if device == 'cuda':
        model = model.cpu()
        torch.cuda.empty_cache()

    if result:
        # Move to final location if needed
        if result != onnx_path:
            import shutil
            shutil.move(str(result), str(onnx_path))
            result = onnx_path

        print(f"\n{'='*80}")
        print(f"SUCCESS: Model converted to ONNX")
        print(f"  Output: {result}")
        print(f"  Size: {result.stat().st_size / (1024**2):.2f} MB")
        print(f"{'='*80}\n")

    return result


def convert_all_checkpoint_components(
    model_path: str,
    output_dir: Optional[str] = None,
    device: str = 'cuda',
    opset_version: int = 17,
    sd_config_path: Optional[str] = None,
) -> dict:
    """
    Convert all components from a Stable Diffusion checkpoint to ONNX.

    Args:
        model_path: Path to the .safetensors/.ckpt file
        output_dir: Output directory for ONNX files
        device: Device to use for conversion
        opset_version: ONNX opset version
        sd_config_path: Optional SD config path

    Returns:
        Dictionary mapping component names to their ONNX paths (or None if failed)
    """
    print(f"\n{'='*80}")
    print(f"Auto-detecting model architecture and converting all components")
    print(f"{'='*80}\n")

    # Detect model version
    version = detect_sd_model_version(model_path)
    print(f"Detected model version: {version}")

    if version == 'unknown':
        print("Warning: Could not auto-detect model version. Assuming SD1.5...")
        version = 'sd1'

    # Components to export
    components = ['unet', 'vae_decoder', 'vae_encoder', 'clip_text']

    results = {}
    for component in components:
        print(f"\n{'-'*80}")
        print(f"Converting component: {component}")
        print(f"{'-'*80}\n")

        try:
            result = convert_model_to_onnx(
                model_path=model_path,
                model_type=component,
                output_dir=output_dir,
                output_name=f"{component}_{Path(model_path).stem}.onnx",
                device=device,
                opset_version=opset_version,
                sd_config_path=sd_config_path,
            )
            results[component] = result

            if result:
                print(f"SUCCESS: {component} exported to {result}")
            else:
                print(f"FAILED: {component} export failed")

        except Exception as e:
            print(f"ERROR exporting {component}: {e}")
            import traceback
            traceback.print_exc()
            results[component] = None

    # Summary
    print(f"\n{'='*80}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*80}\n")

    successful = [k for k, v in results.items() if v is not None]
    failed = [k for k, v in results.items() if v is None]

    if successful:
        print(f"Successfully converted ({len(successful)}/{len(components)}):")
        for comp in successful:
            print(f"  ✓ {comp}: {results[comp]}")

    if failed:
        print(f"\nFailed to convert ({len(failed)}/{len(components)}):")
        for comp in failed:
            print(f"  ✗ {comp}")

    print(f"\n{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AUTO-DETECT: Convert all components from a Stable Diffusion checkpoint (RECOMMENDED)
  python convert_to_onnx.py \\
      --model-path indigoFurryMix_v100Anime.safetensors \\
      --output-dir Cache/test/ \\
      --opset-version 14

  # This will auto-detect the model version and export: UNet, VAE decoder, VAE encoder, and CLIP text encoder

  # Convert only specific component from checkpoint
  python convert_to_onnx.py \\
      --model-path indigoFurryMix_v100Anime.safetensors \\
      --model-type vae_decoder \\
      --output-dir Cache/test/ \\
      --output-name vae_decoder.onnx \\
      --opset-version 14

  # Convert Flux Redux model
  python convert_to_onnx.py \\
      --model-path hf_download/hub/models--lllyasviel--flux_redux_bfl/ \\
      --model-type flux_redux \\
      --output-dir Cache/onnx_models/

  # Convert Stable Diffusion 1.5 from diffusers format (UNet)
  python convert_to_onnx.py \\
      --model-path hf_download/hub/models--runwayml--stable-diffusion-v1-5/ \\
      --model-type unet \\
      --subfolder unet \\
      --output-dir Cache/onnx_sd15

  # Convert with custom SD config
  python convert_to_onnx.py \\
      --model-path checkpoints/v1-5-pruned-emaonly.safetensors \\
      --model-type all \\
      --sd-config-path configs/stable-diffusion/v1-inference.yaml \\
      --output-dir Cache/onnx_sd15

Note:
  - When --model-type is omitted for .safetensors/.ckpt files, it defaults to "all" (exports all components)
  - When --sd-config-path is omitted for checkpoint files, a default SD v1 config is auto-generated
  - Auto-generated configs are cached under Cache/onnx_models/auto_sd_configs/

Supported model types:
  - all: Export all components (UNet, VAE encoder, VAE decoder, CLIP text) - auto-detected for checkpoints
  - unet: UNet diffusion model
  - vae_decoder: VAE decoder (latents -> images)
  - vae_encoder: VAE encoder (images -> latents)
  - clip_text: CLIP text encoder
  - vae: VAE (encoder or decoder, auto-detected)
  - flux_redux: Flux Redux image encoder
  - llama_text: LLaMA text encoder
  - t5_text: T5 text encoder
  - framepack_i2v: FramePackI2V HunyuanVideo transformer
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model directory'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        required=False,
        default=None,
        choices=['flux_redux', 'clip_text', 'llama_text', 't5_text', 'vae', 'vae_encoder', 'vae_decoder', 'unet', 'framepack_i2v', 'all'],
        help='Type of model to convert. Use "all" to export all components from a checkpoint (auto-detected if not specified for .safetensors/.ckpt files)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for ONNX file (default: Cache/onnx_models/)'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Custom output filename (default: auto-generated)'
    )

    parser.add_argument(
        '--subfolder',
        type=str,
        default=None,
        help='Subfolder within model path (for diffusers models)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for conversion (default: cuda)'
    )

    parser.add_argument(
        '--opset-version',
        type=int,
        default=17,
        help='ONNX opset version (default: 17)'
    )

    parser.add_argument(
        '--sd-config-path',
        type=str,
        default=None,
        help='Original Stable Diffusion config .yaml (auto-generated v1 config if omitted for checkpoint files)'
    )

    args = parser.parse_args()

    # Auto-detect model type if not specified
    model_path_obj = Path(args.model_path)
    is_checkpoint = model_path_obj.is_file() and model_path_obj.suffix.lower() in ['.safetensors', '.ckpt']

    if args.model_type is None:
        if is_checkpoint:
            print(f"Auto-detecting model type from checkpoint: {args.model_path}")
            args.model_type = 'all'
        else:
            print("Error: --model-type is required for non-checkpoint files")
            print("Supported types: flux_redux, clip_text, llama_text, t5_text, vae, vae_encoder, vae_decoder, unet, framepack_i2v, all")
            sys.exit(1)

    # Handle "all" option - convert all components from checkpoint
    if args.model_type == 'all':
        if not is_checkpoint:
            print("Error: --model-type=all is only supported for .safetensors/.ckpt checkpoint files")
            sys.exit(1)

        # Print important warning
        print("\n" + "="*80)
        print("⚠️  IMPORTANT: ONNX Conversion is ONE-WAY")
        print("="*80)
        print("\nYou are about to convert your Stable Diffusion checkpoint to ONNX format.")
        print("\nIMPORTANT NOTES:")
        print("  ✓ ONNX files are for inference optimization (ONNX Runtime, TensorRT)")
        print("  ✗ ONNX files CANNOT be converted back to working checkpoints")
        print("  ✗ Weight names and structure will be different from the original")
        print("  ✗ Resulting files will NOT work with SD WebUI or ComfyUI directly")
        print("\nMAKE SURE TO:")
        print("  • Keep your original .safetensors checkpoint file")
        print("  • Only use ONNX files for optimized inference")
        print("  • Do NOT delete the original checkpoint")
        print("\n" + "="*80 + "\n")

        response = input("Do you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Conversion cancelled.")
            sys.exit(0)

        results = convert_all_checkpoint_components(
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=args.device,
            opset_version=args.opset_version,
            sd_config_path=args.sd_config_path,
        )

        # Check if at least one component succeeded
        if any(v is not None for v in results.values()):
            print("Conversion completed with some successes!")
            sys.exit(0)
        else:
            print("\nAll conversions failed!")
            sys.exit(1)

    # Convert single model
    result = convert_model_to_onnx(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        output_name=args.output_name,
        device=args.device,
        opset_version=args.opset_version,
        subfolder=args.subfolder,
        sd_config_path=args.sd_config_path,
    )

    if result is None:
        print("\nConversion failed!")
        sys.exit(1)

    print("Conversion completed successfully!")
    sys.exit(0)


if __name__ == '__main__':
    main()
