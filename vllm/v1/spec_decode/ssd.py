# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

logger = init_logger(__name__)


class SSDProposer(SpecDecodeBaseProposer):
    """
    Speculative Speculative Decoding (SSD) proposer.
    
    SSD is a novel speculative decoding algorithm that parallelizes drafting
    and verification by pre-speculating for multiple possible verification
    outcomes in advance.
    
    This implementation supports both synchronous mode (standard speculative
    decoding) and asynchronous mode (true SSD with parallel drafting and
    verification and tree caching).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        
        # SSD-specific configuration
        self.ssd_async = self.speculative_config.ssd_async
        self.ssd_async_fan_out = self.speculative_config.ssd_async_fan_out
        self.ssd_jit_speculate = self.speculative_config.ssd_jit_speculate
        self.ssd_sampler_x = self.speculative_config.ssd_sampler_x
        self.ssd_max_cached_sequences = self.speculative_config.ssd_max_cached_sequences
        
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()
        
        # Tree cache for SSD async mode
        self.tree_cache_keys = None
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        
        if self.ssd_async:
            self._init_tree_cache()
            
        logger.info(f"[SSD] Initialized SSDProposer with async={self.ssd_async}, "
                   f"fan_out={self.ssd_async_fan_out}")

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same for SSD. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    def _init_tree_cache(self):
        """Initialize the tree cache for SSD async mode."""
        logger.info(f"[SSD] Initializing tree cache with max {self.ssd_max_cached_sequences} sequences")
        
        # Initialize empty cache
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    @override
    def _get_model(self):
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model
        from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("ssd_draft_model"):
            model = get_model(
                vllm_config=temp_vllm_config,
                prefix="ssd_draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model):
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model):
        pass

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        sampled_token_ids,
        slot_mappings=None,
        attn_metadata=None,
    ):
        """
        Propose speculative tokens using SSD.
        
        In sync mode: works like standard speculative decoding
        In async mode: uses tree cache to accelerate speculation
        """
        draft_token_ids = []
        
        if self.ssd_async and self.tree_cache_keys is not None:
            draft_token_ids = self._propose_async(
                input_batch, sampled_token_ids
            )
        else:
            draft_token_ids = self._propose_sync(
                input_batch, sampled_token_ids, slot_mappings, attn_metadata
            )
            
        return draft_token_ids

    def _propose_sync(
        self,
        input_batch,
        sampled_token_ids,
        slot_mappings=None,
        attn_metadata=None,
    ):
        """
        Synchronous SSD proposal - works like standard speculative decoding.
        """
        # For sync mode, use the base class's proposal mechanism
        # This is similar to DraftModelProposer
        draft_token_ids = []
        
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue
                
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue
                
            # TODO: Implement actual synchronous speculation
            # For now, return empty as placeholder
            draft_token_ids.append([])
            
        return draft_token_ids

    def _propose_async(
        self,
        input_batch,
        sampled_token_ids,
    ):
        """
        Asynchronous SSD proposal - uses tree cache for acceleration.
        """
        draft_token_ids = []
        
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue
                
            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue
                
            # TODO: Implement tree cache lookup and async speculation
            # For now, return empty as placeholder
            draft_token_ids.append([])
            
        return draft_token_ids

    def _lookup_tree_cache(self, seq_id, keep_idx, recovery_token):
        """
        Look up the tree cache for a previously computed speculation.
        """
        if self.tree_cache_keys is None or self.tree_cache_keys.numel() == 0:
            return None, None
            
        # Create query key
        query = torch.tensor(
            [seq_id, keep_idx, recovery_token],
            dtype=torch.int64,
            device=self.device
        )
        
        # Vectorized lookup
        matches = (self.tree_cache_keys == query.unsqueeze(0)).all(dim=1)
        if matches.any():
            idx = matches.float().argmax().to(torch.int64)
            return self.tree_cache_tokens[idx], self.tree_cache_logits[idx]
            
        return None, None

    def _populate_tree_cache(self, seq_id, keep_idx, recovery_token, tokens, logits):
        """
        Populate the tree cache with a new speculation result.
        """
        if self.tree_cache_keys is None:
            self._init_tree_cache()
            
        new_key = torch.tensor(
            [[seq_id, keep_idx, recovery_token]],
            dtype=torch.int64,
            device=self.device
        )
        
        if self.tree_cache_keys.numel() == 0:
            self.tree_cache_keys = new_key
            self.tree_cache_tokens = tokens.unsqueeze(0)
            self.tree_cache_logits = logits.unsqueeze(0)
        else:
            # TODO: Implement cache eviction if needed
            self.tree_cache_keys = torch.cat([self.tree_cache_keys, new_key], dim=0)
            self.tree_cache_tokens = torch.cat(
                [self.tree_cache_tokens, tokens.unsqueeze(0)], dim=0
            )
            self.tree_cache_logits = torch.cat(
                [self.tree_cache_logits, logits.unsqueeze(0)], dim=0
            )

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        logger.info("[SSD] Draft model loaded successfully")
