"""Documented OpenPI checkpoint aliases, restricted to the PI0.5 family."""

import torch


def resolve_candidates(name, candidates, checkpoint):
    aliases = {
        ('paligemma_with_expert.paligemma.model.'
         'language_model.embed_tokens.weight'),
        'paligemma_with_expert.paligemma.lm_head.weight',
    }
    if name == 'llm_backbone.embed_tokens.weight' and set(
            candidates) == aliases:
        if not torch.equal(checkpoint[candidates[0]],
                           checkpoint[candidates[1]]):
            raise ValueError('shared embedding aliases disagree')
        return candidates[:1]
    # Flux's unused expert vocabulary embedding is absent in OpenPI;
    # its matching LM head is the only supported substitute.
    if not candidates and name == 'llm_expert.embed_tokens.weight':
        alias = 'paligemma_with_expert.gemma_expert.lm_head.weight'
        if alias in checkpoint:
            return [alias]
    return candidates
