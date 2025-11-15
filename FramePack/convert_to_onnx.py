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


def convert_model_to_onnx(
    model_path: str,
    model_type: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    device: str = 'cuda',
    opset_version: int = 17,
    subfolder: Optional[str] = None,
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
        'unet': load_diffusers_model,
    }

    sample_input_generators = {
        'flux_redux': get_sample_inputs_flux_redux,
        'clip_text': get_sample_inputs_text_encoder,
        'llama_text': get_sample_inputs_text_encoder,
        't5_text': get_sample_inputs_text_encoder,
        'vae_encoder': get_sample_inputs_vae_encoder,
        'vae_decoder': get_sample_inputs_vae_decoder,
        'vae': get_sample_inputs_vae_encoder,
        'unet': get_sample_inputs_vae_encoder,
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
    }

    # Load model
    loader = model_loaders.get(model_type)
    if loader is None:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Supported types: {', '.join(model_loaders.keys())}")
        return None

    try:
        if subfolder and model_type in ['vae', 'unet']:
            model, config = load_diffusers_model(model_path, subfolder)
        else:
            model, config = loader(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            'image_embeddings': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'},
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
        model_name = f"{model_type}_{Path(model_path).name}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Flux Redux model
  python convert_to_onnx.py \\
      --model-path hf_download/hub/models--lllyasviel--flux_redux_bfl/ \\
      --model-type flux_redux \\
      --output-dir Cache/onnx_models/

  # Convert CLIP text encoder
  python convert_to_onnx.py \\
      --model-path hf_download/hub/models--openai--clip-vit-large-patch14/ \\
      --model-type clip_text \\
      --output-name clip_text_encoder.onnx

  # Convert VAE from diffusers model
  python convert_to_onnx.py \\
      --model-path hf_download/hub/models--stabilityai--stable-diffusion-xl-base-1.0/ \\
      --model-type vae \\
      --subfolder vae

Supported model types:
  - flux_redux: Flux Redux image encoder
  - clip_text: CLIP text encoder
  - llama_text: LLaMA text encoder
  - t5_text: T5 text encoder
  - vae: VAE (encoder or decoder)
  - unet: UNet model
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
        required=True,
        choices=['flux_redux', 'clip_text', 'llama_text', 't5_text', 'vae', 'vae_encoder', 'vae_decoder', 'unet'],
        help='Type of model to convert'
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

    args = parser.parse_args()

    # Convert model
    result = convert_model_to_onnx(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        output_name=args.output_name,
        device=args.device,
        opset_version=args.opset_version,
        subfolder=args.subfolder,
    )

    if result is None:
        print("\nConversion failed!")
        sys.exit(1)

    print("Conversion completed successfully!")
    sys.exit(0)


if __name__ == '__main__':
    main()
