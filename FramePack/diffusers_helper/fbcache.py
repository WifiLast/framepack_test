from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import torch


@dataclasses.dataclass
class FirstBlockCacheConfig:
    enabled: bool = True
    threshold: float = 0.12
    verbose: bool = False


class FirstBlockCache:
    """
    Lightweight first-block cache for transformer-style models.

    Reuses the expensive \"rest of network\" computation when the residual of the
    first block is close to the previously computed one. We store the residual of
    the remaining blocks and simply add it back when a cache hit occurs.
    """

    def __init__(self, config: FirstBlockCacheConfig | None = None):
        self.config = config or FirstBlockCacheConfig()
        self.prev_first_residual: Optional[torch.Tensor] = None
        self.hidden_states_residual: Optional[torch.Tensor] = None
        self.encoder_hidden_states_residual: Optional[torch.Tensor] = None
        self._last_diff = None

    def reset(self):
        self.prev_first_residual = None
        self.hidden_states_residual = None
        self.encoder_hidden_states_residual = None
        self._last_diff = None

    def update_config(self, config: FirstBlockCacheConfig):
        self.config = config
        if not self.config.enabled:
            self.reset()

    def maybe_run(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
        attention_mask,
        rope_freqs,
        transformer_blocks: Sequence[torch.nn.Module],
        single_transformer_blocks: Sequence[torch.nn.Module],
        grad_ckpt_fn,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.hidden_states_residual is not None
            and self.hidden_states_residual.device != hidden_states.device
        ):
            self.reset()
        if not self.config.enabled or not transformer_blocks:
            return self._run_full(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_mask=attention_mask,
                rope_freqs=rope_freqs,
                transformer_blocks=transformer_blocks,
                single_transformer_blocks=single_transformer_blocks,
                grad_ckpt_fn=grad_ckpt_fn,
            )

        original_states = hidden_states
        encoder_original = encoder_hidden_states

        first_block = transformer_blocks[0]
        hidden_states, encoder_hidden_states = grad_ckpt_fn(
            first_block,
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            rope_freqs,
        )

        first_residual = (hidden_states - original_states).detach()
        reference_hidden_states = hidden_states.detach()
        reference_encoder_states = encoder_hidden_states.detach() if encoder_hidden_states is not None else None
        can_use_cache = self._can_use_cache(first_residual)

        if can_use_cache and self.hidden_states_residual is not None:
            hidden_states = hidden_states + self.hidden_states_residual
            hidden_states = hidden_states.contiguous()
            if (
                encoder_hidden_states is not None
                and self.encoder_hidden_states_residual is not None
            ):
                encoder_hidden_states = encoder_hidden_states + self.encoder_hidden_states_residual
                encoder_hidden_states = encoder_hidden_states.contiguous()
            return hidden_states, encoder_hidden_states

        # cache miss -> run remaining blocks and store residuals
        hidden_states, encoder_hidden_states = self._run_remaining(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            attention_mask=attention_mask,
            rope_freqs=rope_freqs,
            transformer_blocks=transformer_blocks[1:],
            single_transformer_blocks=single_transformer_blocks,
            grad_ckpt_fn=grad_ckpt_fn,
        )

        self.hidden_states_residual = (hidden_states - reference_hidden_states).detach()
        if (
            encoder_hidden_states is not None
            and reference_encoder_states is not None
        ):
            self.encoder_hidden_states_residual = (encoder_hidden_states - reference_encoder_states).detach()
        else:
            self.encoder_hidden_states_residual = None

        return hidden_states, encoder_hidden_states

    def _run_full(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
        attention_mask,
        rope_freqs,
        transformer_blocks: Sequence[torch.nn.Module],
        single_transformer_blocks: Sequence[torch.nn.Module],
        grad_ckpt_fn,
    ):
        hidden_states, encoder_hidden_states = self._run_remaining(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            rope_freqs,
            transformer_blocks,
            single_transformer_blocks,
            grad_ckpt_fn,
        )
        return hidden_states, encoder_hidden_states

    def _run_remaining(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
        attention_mask,
        rope_freqs,
        transformer_blocks: Sequence[torch.nn.Module],
        single_transformer_blocks: Sequence[torch.nn.Module],
        grad_ckpt_fn,
    ):
        for block in transformer_blocks:
            hidden_states, encoder_hidden_states = grad_ckpt_fn(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs,
            )
        for block in single_transformer_blocks:
            hidden_states, encoder_hidden_states = grad_ckpt_fn(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs,
            )
        return hidden_states, encoder_hidden_states

    def _can_use_cache(self, new_residual: torch.Tensor) -> bool:
        if self.prev_first_residual is None:
            self.prev_first_residual = new_residual
            return False

        denom = torch.clamp(self.prev_first_residual.abs().mean(), min=1e-6)
        diff = (new_residual - self.prev_first_residual).abs().mean()
        ratio = (diff / denom).item()
        self._last_diff = ratio

        self.prev_first_residual = new_residual
        if ratio < self.config.threshold and self.hidden_states_residual is not None:
            if self.config.verbose:
                print(f"[FBCache] hit diff={ratio:.4f} (threshold={self.config.threshold:.4f})")
            return True

        if self.config.verbose:
            print(f"[FBCache] miss diff={ratio:.4f} (threshold={self.config.threshold:.4f})")
        return False

    @staticmethod
    def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().cpu()

    def state_dict(self):
        return {
            "config": dataclasses.asdict(self.config) if self.config else None,
            "prev_first_residual": self._clone_tensor(self.prev_first_residual),
            "hidden_states_residual": self._clone_tensor(self.hidden_states_residual),
            "encoder_hidden_states_residual": self._clone_tensor(self.encoder_hidden_states_residual),
            "_last_diff": self._last_diff,
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        config_data = state_dict.get("config")
        if config_data:
            self.config = FirstBlockCacheConfig(**config_data)
        self.prev_first_residual = state_dict.get("prev_first_residual")
        self.hidden_states_residual = state_dict.get("hidden_states_residual")
        self.encoder_hidden_states_residual = state_dict.get("encoder_hidden_states_residual")
        self._last_diff = state_dict.get("_last_diff")
