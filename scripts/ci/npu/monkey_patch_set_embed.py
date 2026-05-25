"""Runtime monkey-patch for missing set_embed/get_embed on DeepseekV2ForCausalLM.

This is needed because some sglang model classes (like DeepseekV2ForCausalLM)
do not implement set_embed/get_embed methods required by EAGLE3 speculative decoding.
We patch at runtime rather than modifying sglang source code.
"""

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
import torch


def _set_embed(self, embed):
    del self.model.embed_tokens.weight
    self.model.embed_tokens.weight = embed
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _get_embed(self):
    return self.model.embed_tokens.weight


DeepseekV2ForCausalLM.set_embed = _set_embed
DeepseekV2ForCausalLM.get_embed = _get_embed
print('Applied monkey-patch for set_embed/get_embed on DeepseekV2ForCausalLM')