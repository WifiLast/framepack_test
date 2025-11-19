"""
Utilities that learn lightweight surrogates for DiT blocks.

- `HiddenStateRelationshipTrainer` keeps the original online learner.
- `DiTTimestepResidualTrainer` freezes the DiT block and learns only the timestep-modulation residual.
- `DiTTimestepModulationTrainer` replaces the AdaLN/FiLM timestep MLP with a compact predictor over timesteps.
"""

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformerBlock,
)


class HiddenStateRelationshipTrainer(nn.Module):
    """Original lightweight MLP used for online residual learning."""

    def __init__(self, hidden_dim: int, lr: float = 1e-4, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._device = device

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        ).to(self._device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler() if "cuda" in self._device else None

    def to(self, device):
        self._device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler() if "cuda" in str(device) else None
        return self

    @property
    def device(self):
        return self._device

    def train_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        self.model.train()
        input_tensor = input_tensor.to(self.device).detach()
        target_tensor = target_tensor.to(self.device).detach()

        self.optimizer.zero_grad()

        if self.scaler:
            with torch.cuda.amp.autocast():
                prediction = self.model(input_tensor)
                loss = self.loss_fn(prediction, target_tensor)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            prediction = self.model(input_tensor)
            loss = self.loss_fn(prediction, target_tensor)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not isinstance(state_dict, dict):
            print("Warning: Relationship trainer state_dict is invalid, skipping load.")
            return

        self.lr = state_dict.get("lr", self.lr)
        if self.hidden_dim != state_dict.get("hidden_dim"):
            print(
                "Warning: Hidden dimension mismatch for relationship trainer. "
                f"Expected {self.hidden_dim}, got {state_dict.get('hidden_dim')}. Re-initializing model."
            )
            self.hidden_dim = state_dict.get("hidden_dim", self.hidden_dim)
            self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            ).to(self.device)

        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.model.to(self.device)
        print("Loaded relationship trainer state.")


class DiTTimestepResidualPredictor(nn.Module):
    """Predicts Δh using a compact bottleneck network."""

    def __init__(
        self,
        hidden_dim: int,
        temb_dim: int,
        bottleneck_dim: int = 512,
        num_hidden_layers: int = 2,
        low_rank_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.t_proj = nn.Linear(temb_dim, bottleneck_dim)

        layers = []
        for _ in range(max(1, num_hidden_layers)):
            layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.core = nn.Sequential(*layers)

        if low_rank_dim is not None and 0 < low_rank_dim < bottleneck_dim:
            self.low_rank_down = nn.Linear(bottleneck_dim, low_rank_dim)
            self.low_rank_up = nn.Linear(low_rank_dim, bottleneck_dim)
        else:
            self.low_rank_down = None
            self.low_rank_up = None

        self.out_proj = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, h_in: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if h_in.ndim != 3:
            raise ValueError(f"h_in must be (B, N, C), got {h_in.shape}")
        if temb.ndim != 2:
            raise ValueError(f"temb must be (B, D), got {temb.shape}")

        projected_tokens = self.in_proj(h_in)
        projected_temporal = self.t_proj(temb).unsqueeze(1).expand_as(projected_tokens)
        combined = torch.silu(projected_tokens + projected_temporal)

        hidden = self.core(combined)
        if self.low_rank_down is not None and self.low_rank_up is not None:
            hidden = self.low_rank_up(self.low_rank_down(hidden))

        residual = self.out_proj(hidden)
        return residual


class DiTTimestepResidualTrainer(nn.Module):
    """Freeze a DiT block and learn only the timestep-modulation residual Δh."""

    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        temb_dim: int,
        *,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        predictor_bottleneck: int = 512,
        predictor_layers: int = 2,
        predictor_low_rank: Optional[int] = 128,
    ) -> None:
        super().__init__()
        self.block = block
        self.hidden_dim = hidden_dim
        self.temb_dim = temb_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.block.eval()
        for param in self.block.parameters():
            param.requires_grad_(False)

        self.predictor = DiTTimestepResidualPredictor(
            hidden_dim=hidden_dim,
            temb_dim=temb_dim,
            bottleneck_dim=predictor_bottleneck,
            num_hidden_layers=predictor_layers,
            low_rank_dim=predictor_low_rank,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> float:
        h_in, delta = self._collect_residual(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            image_rotary_emb=image_rotary_emb,
        )
        prediction = self.predictor(h_in, temb.to(self.device))
        loss = self.loss_fn(prediction, delta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def approximate_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_base, enc_base = self._forward_without_timestep(
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            image_rotary_emb=image_rotary_emb,
        )
        h_in = hidden_states.detach().to(self.device)
        residual = self.predictor(h_in, temb.to(self.device))
        if residual.device != h_base.device:
            residual = residual.to(h_base.device)
        approx_hidden = h_base + residual
        return approx_hidden, enc_base

    @torch.no_grad()
    def _collect_residual(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_in = hidden_states.detach().to(self.device)
        temb = temb.detach()

        block_kwargs: Dict[str, Any] = {}
        if isinstance(self.block, HunyuanVideoTransformerBlock):
            if attention_mask is not None:
                block_kwargs["attention_mask"] = attention_mask
            if freqs_cis is not None:
                block_kwargs["freqs_cis"] = freqs_cis
        elif isinstance(self.block, HunyuanVideoSingleTransformerBlock):
            if attention_mask is not None:
                block_kwargs["attention_mask"] = attention_mask
            if image_rotary_emb is not None:
                block_kwargs["image_rotary_emb"] = image_rotary_emb
        else:
            raise TypeError(f"Unsupported block type for residual collection: {type(self.block)}")

        full_hidden, _ = self.block(
            hidden_states.detach(),
            encoder_hidden_states.detach(),
            temb,
            **block_kwargs,
        )
        base_hidden, _ = self._forward_without_timestep(
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            image_rotary_emb=image_rotary_emb,
        )
        delta = (full_hidden - base_hidden).detach().to(self.device)
        return h_in, delta

    def _forward_without_timestep(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.block
        if isinstance(block, HunyuanVideoTransformerBlock):
            return self._forward_hunyuan_transformer_block_no_timestep(
                block,
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
            )
        if isinstance(block, HunyuanVideoSingleTransformerBlock):
            return self._forward_hunyuan_single_block_no_timestep(
                block,
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
        raise TypeError(f"Unsupported block type for timestep residual trainer: {type(block)}")

    @staticmethod
    @torch.no_grad()
    def _forward_hunyuan_transformer_block_no_timestep(
        block: HunyuanVideoTransformerBlock,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states = block.norm1.norm(hidden_states)
        norm_encoder_hidden_states = block.norm1_context.norm(encoder_hidden_states)

        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_hidden_states = block.norm2(hidden_states)
        norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)

        ff_output = block.ff(norm_hidden_states)
        context_ff_output = block.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + ff_output
        encoder_hidden_states = encoder_hidden_states + context_ff_output
        return hidden_states, encoder_hidden_states

    @staticmethod
    @torch.no_grad()
    def _forward_hunyuan_single_block_no_timestep(
        block: HunyuanVideoSingleTransformerBlock,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        full_hidden = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = full_hidden

        norm_hidden = block.norm.norm(full_hidden)
        mlp_hidden = block.act_mlp(block.proj_mlp(norm_hidden))
        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden[:, :-text_seq_length, :],
            norm_hidden[:, -text_seq_length:, :],
        )

        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        projected = block.proj_out(torch.cat([attn_output, mlp_hidden], dim=2))
        full_hidden = projected + residual
        hidden_states, encoder_hidden_states = (
            full_hidden[:, :-text_seq_length, :],
            full_hidden[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class DiTTimestepModulationPredictor(nn.Module):
    """Predict gamma(t) and beta(t) directly from timestep embeddings."""

    def __init__(self, temb_dim: int, mod_dim: int, hidden_dim: int = 512, layers: int = 2) -> None:
        super().__init__()
        net = []
        in_dim = temb_dim
        for i in range(max(1, layers)):
            out_dim = hidden_dim if i < layers - 1 else mod_dim * 2
            net.append(nn.Linear(in_dim, out_dim))
            if i < layers - 1:
                net.append(nn.SiLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*net)

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if temb.ndim != 2:
            raise ValueError("temb must be (B, D)")
        output = self.network(temb)
        gamma_hat, beta_hat = output.chunk(2, dim=-1)
        return gamma_hat, beta_hat


class DiTTimestepModulationTrainer(nn.Module):
    """Learns to reproduce the AdaLN/AdaLayerNorm modulation parameters."""

    def __init__(
        self,
        block: nn.Module,
        temb_dim: int,
        mod_dim: int,
        *,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        predictor_hidden: int = 512,
        predictor_layers: int = 2,
    ) -> None:
        super().__init__()
        self.block = block
        self.temb_dim = temb_dim
        self.mod_dim = mod_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.block.eval()
        for param in self.block.parameters():
            param.requires_grad_(False)

        self.predictor = DiTTimestepModulationPredictor(
            temb_dim=temb_dim,
            mod_dim=mod_dim,
            hidden_dim=predictor_hidden,
            layers=predictor_layers,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, temb: torch.Tensor) -> float:
        gamma, beta = self._extract_modulation(temb)
        gamma_hat, beta_hat = self.predictor(temb.to(self.device))
        loss = self.loss_fn(gamma_hat, gamma) + self.loss_fn(beta_hat, beta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_hat, beta_hat = self.predictor(temb.to(self.device))
        return gamma_hat, beta_hat

    @torch.no_grad()
    def _extract_modulation(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = temb.to(self.device)
        if isinstance(self.block, HunyuanVideoTransformerBlock):
            module = self.block.norm1
        elif isinstance(self.block, HunyuanVideoSingleTransformerBlock):
            module = self.block.norm
        else:
            raise TypeError(f"Unsupported block for modulation extraction: {type(self.block)}")

        emb = module.linear(module.silu(temb.unsqueeze(-2)))
        if isinstance(self.block, HunyuanVideoTransformerBlock):
            shift_msa, scale_msa, *_ = emb.chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, _ = emb.chunk(3, dim=-1)
        gamma = (1 + scale_msa).squeeze(1)
        beta = shift_msa.squeeze(1)
        return gamma.detach(), beta.detach()
