# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn, Tensor

from kbcr.models.base import BaseLatentFeatureModel

from typing import Tuple, Optional
from profilehooks import profile

logger = logging.getLogger(__name__)


class TransE(BaseLatentFeatureModel):
    def __init__(self,
                 entity_embeddings: nn.Embedding,
                 predicate_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.embedding_size = self.entity_embeddings.weight.shape[1]

    @profile(immediate=True)
    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B, E]
        res_vector = arg1+rel-arg2

        # [B] Tensor
        res = torch.norm(res_vector, dim=1)

        # [B] Tensor
        return res
    
    @profile(immediate=True)
    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        emb = self.entity_embeddings.weight
        p_emb = self.predicate_embeddings.weight

        # [B] Tensor
        score_sp = score_po = None

        if arg1 is not None:
            arg1_expand = Tensor(arg1+rel)
            arg1_expand.unsqueeze_(-1)
            arg1_expand = arg1_expand.expand(arg1_expand.shape[0], arg1_expand.shape[1], emb.shape[0])
            emb_expand = Tensor(p_emb)
            emb_expand.unsqueeze_(-1)
            emb_expand = emb_expand.expand(emb_expand.shape[0], emb_expand.shape[1], arg1_expand.shape[0])
            emb_expand = emb_expand.permute(2, 1, 0)


            score_sp = torch.norm(arg1_expand-emb_expand, dim=1)

        if arg2 is not None:
            arg2_expand = Tensor(rel-arg2)
            arg2_expand.unsqueeze_(-1)
            arg2_expand = arg2_expand.expand(arg2_expand.shape[0], arg2_expand.shape[1], emb.shape[0])
            emb_expand = Tensor(p_emb)
            emb_expand.unsqueeze_(-1)
            emb_expand = emb_expand.expand(emb_expand.shape[0], emb_expand.shape[1], arg2_expand.shape[0])
            emb_expand = emb_expand.permute(2, 1, 0)


            score_po = torch.norm(emb_expand-arg2_expand, dim=1)

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        vec_real = embedding_vector[:, :self.embedding_size]
        vec_img = embedding_vector[:, self.embedding_size:]
        return torch.sqrt(vec_real ** 2 + vec_img ** 2)
