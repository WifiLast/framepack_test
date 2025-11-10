import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


@dataclass
class SimilarityCacheConfig:
    enabled: bool = True
    threshold: float = 0.9
    max_skip: int = 1
    max_entries: int = 16
    use_faiss: bool = False
    max_age: int = 32
    verbose: bool = False


class LearnableProjector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cached_hidden: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        projected = self.linear(cached_hidden)
        similarity = similarity.clamp(0.0, 1.0)
        while similarity.dim() < projected.dim():
            similarity = similarity.unsqueeze(-1)
        return cached_hidden + similarity * (projected - cached_hidden)


class BlockCacheEntry:
    def __init__(self, key: torch.Tensor, hidden: torch.Tensor, encoder: Optional[torch.Tensor], step: int):
        self.key = F.normalize(key.float(), dim=-1)
        self.hidden = hidden.detach().cpu()
        self.encoder = encoder.detach().cpu() if encoder is not None else None
        self.step = step


class BlockSimilarityCache:
    def __init__(self, key_dim: int, max_entries: int = 16, use_faiss: bool = False, max_age: int = 32):
        self.key_dim = key_dim
        self.max_entries = max_entries
        self.max_age = max_age
        self.entries: list[BlockCacheEntry] = []
        self.use_faiss = use_faiss and _HAS_FAISS
        self._faiss_index = None
        if self.use_faiss:
            self._faiss_index = faiss.IndexFlatIP(key_dim)

    def _rebuild_index(self):
        if not self.use_faiss:
            return
        self._faiss_index.reset()
        if not self.entries:
            return
        keys = torch.stack([entry.key for entry in self.entries]).numpy()
        self._faiss_index.add(keys)

    def lookup(self, key: torch.Tensor) -> Optional[Tuple[BlockCacheEntry, float]]:
        if not self.entries:
            return None
        key = F.normalize(key.float(), dim=-1)
        if self.use_faiss and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(key.cpu().numpy()[None, :], 1)
            idx = int(indices[0][0])
            if idx < 0 or idx >= len(self.entries):
                return None
            sim = float(distances[0][0])
            return self.entries[idx], sim
        stacked = torch.stack([entry.key for entry in self.entries])
        sims = torch.mv(stacked, key.cpu())
        max_sim, idx = torch.max(sims, dim=0)
        return self.entries[int(idx)], float(max_sim.item())

    def update(self, key: torch.Tensor, hidden: torch.Tensor, encoder: Optional[torch.Tensor], step: int):
        entry = BlockCacheEntry(key, hidden, encoder, step)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        if self.use_faiss:
            self._rebuild_index()

    def prune(self, current_step: int):
        if self.max_age <= 0 or not self.entries:
            return
        keep = [entry for entry in self.entries if current_step - entry.step <= self.max_age]
        if len(keep) != len(self.entries):
            self.entries = keep
            if self.use_faiss:
                self._rebuild_index()

    def clear(self):
        self.entries.clear()
        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.reset()


class SimilarityCacheManager:
    def __init__(
        self,
        num_dual_blocks: int,
        num_single_blocks: int,
        hidden_dim: int,
        config: SimilarityCacheConfig,
    ):
        self.config = config
        self.global_step = 0
        self.dual_caches = [
            BlockSimilarityCache(hidden_dim, config.max_entries, config.use_faiss, config.max_age)
            for _ in range(num_dual_blocks)
        ]
        self.single_caches = [
            BlockSimilarityCache(hidden_dim, config.max_entries, config.use_faiss, config.max_age)
            for _ in range(num_single_blocks)
        ]

    def step(self):
        self.global_step += 1

    def get(self, block_type: str, block_id: int) -> BlockSimilarityCache:
        if block_type == "dual":
            return self.dual_caches[block_id]
        return self.single_caches[block_id]

    def prune(self):
        for cache in self.dual_caches + self.single_caches:
            cache.prune(self.global_step)

    def clear(self):
        for cache in self.dual_caches + self.single_caches:
            cache.clear()
