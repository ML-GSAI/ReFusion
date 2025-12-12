import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, List, Tuple, Dict, Any

class DiffusionDynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None):
        super().__init__(num_hidden_layers)

    def full_update(
        self,
        new_kv: Tuple,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for i, (key, val) in enumerate(new_kv):
            self.key_cache[i] = torch.cat([self.key_cache[i], key], dim=-2)
            self.value_cache[i] = torch.cat([self.value_cache[i], val], dim=-2)
    
    def select_partial(
        self,
        indices: torch.Tensor,
    ):
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i][:, :, indices, :]
            self.value_cache[i] = self.value_cache[i][:, :, indices, :]
    
    def batch_select_minibatch(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:indices, ...]