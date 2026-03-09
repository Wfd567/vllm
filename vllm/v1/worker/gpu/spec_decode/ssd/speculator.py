# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class SSDSpeculator:
    """
    GPU-side SSD (Speculative Speculative Decoding) speculator.
    
    This class handles the actual GPU execution of SSD, including:
    - Tree cache management
    - Draft model execution
    - Speculative token generation
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        # SSD-specific configuration
        self.ssd_async = self.speculative_config.ssd_async
        self.ssd_async_fan_out = self.speculative_config.ssd_async_fan_out
        self.ssd_jit_speculate = self.speculative_config.ssd_jit_speculate
        self.ssd_sampler_x = self.speculative_config.ssd_sampler_x
        
        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        
        # Tree cache for SSD
        if self.ssd_async:
            self._init_tree_cache()

        logger.info(f"[SSD] Initialized SSDSpeculator (async={self.ssd_async})")

    def _init_tree_cache(self):
        """Initialize the tree cache for SSD async mode."""
        logger.info(f"[SSD] Initializing tree cache on GPU")
        
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    def load_model(self, target_model: nn.Module) -> None:
        """Load the SSD draft model."""
        from vllm.model_executor.model_loader import get_model
        from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("ssd_draft_model"):
            self.model = get_model(
                vllm_config=temp_vllm_config,
                prefix="ssd_draft_model",
            )
        logger.info("[SSD] Draft model loaded successfully")

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        block_tables: BlockTables,
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups = attn_groups
        self.block_tables = block_tables

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )
        return hidden_states

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> torch.Tensor:
        """
        Propose speculative tokens using SSD.
        
        Args:
            input_batch: Input batch containing the current requests
            attn_metadata: Attention metadata
            slot_mappings: Slot mappings for KV cache
            last_hidden_states: Last hidden states from target model
            aux_hidden_states: Auxiliary hidden states (for EAGLE3)
            num_sampled: Number of sampled tokens
            num_rejected: Number of rejected tokens
            last_sampled: Last sampled tokens
            next_prefill_tokens: Next prefill tokens
            temperature: Sampling temperatures
            seeds: Sampling seeds
            num_tokens_across_dp: Number of tokens across DP
            dummy_run: Whether this is a dummy run
            skip_attn_for_dummy_run: Whether to skip attention for dummy run
            
        Returns:
            Draft tokens tensor of shape [num_reqs, num_speculative_steps]
        """
        num_reqs = input_batch.num_reqs
        
        if num_reqs == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device)
            
        # For now, return empty draft tokens as placeholder
        # This will be implemented with full SSD logic
        draft_tokens = torch.zeros(
            (num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=self.device,
        )
        
        return draft_tokens

    def capture_model(self):
        """Capture the draft model for CUDA graph optimization."""
        # TODO: Implement CUDA graph capture for SSD
        pass
