# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from kbcr.clutrr.models.kb import NeuralKB
from kbcr.reformulators import BaseReformulator
from kbcr.reformulators import GNTPReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class Hoppy(nn.Module):
    def __init__(self,
                 model: NeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 k: int = 10,
                 depth: int = 0,
                 tnorm_name: str = 'min',
                 R: Optional[int] = None):
        super().__init__()

        self.model: NeuralKB = model
        self.k = k

        self.depth = depth
        assert self.depth >= 0

        self.tnorm_name = tnorm_name
        assert self.tnorm_name in {'min', 'prod'}

        self.R = R

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

        logger.info(f'Hoppy(k={k}, depth={depth}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')

    def _tnorm(self, x: Tensor, y: Tensor):
        return x * y if self.tnorm_name == 'prod' else torch.min(x, y)

    def r_hop(self,
              rel: Tensor,
              arg1: Optional[Tensor],
              arg2: Optional[Tensor],
              facts: List[Tensor],
              entity_embeddings: Tensor,
              depth: int) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)
        assert depth >= 0

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        scores_sp, scores_po = self.r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=depth)
        scores = scores_sp if arg2 is None else scores_po

        k = min(self.k, scores.shape[1])

        # [B, K], [B, K]
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)
        # [B, K, E]
        # z_emb = entity_embeddings(z_indices)
        z_emb = F.embedding(z_indices, entity_embeddings)

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        return z_scores, z_emb

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              facts: List[Tensor],
              entity_embeddings: Tensor) -> Tensor:
        res = self.r_score(rel, arg1, arg2, facts, entity_embeddings, depth=self.depth)
        return res

    def r_score(self,
                rel: Tensor,
                arg1: Tensor,
                arg2: Tensor,
                facts: List[Tensor],
                entity_embeddings: Tensor,
                depth: int) -> Tensor:
        res = None
        for d in range(depth + 1):
            scores = self.depth_r_score(rel, arg1, arg2, facts, entity_embeddings, depth=d)
            res = scores if res is None else torch.max(res, scores)
        return res

    def depth_r_score(self,
                      rel: Tensor,
                      arg1: Tensor,
                      arg2: Tensor,
                      facts: List[Tensor],
                      entity_embeddings: Tensor,
                      depth: int) -> Tensor:
        assert depth >= 0

        if depth == 0:
            return self.model.score(rel, arg1, arg2, facts)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

            # import sys
            # sys.exit(0)

            # mask = torch.zeros_like(batch_rules_scores).scatter_(1, indices, torch.ones_like(topk))

        # for hops_generator, is_reversed in self.hops_lst:
        # for rule_idx, (hops_generator, is_reversed) in enumerate(self.hops_lst):
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            sources, scores = arg1, None

            # XXX
            prior = hops_generator.prior(rel)
            if prior is not None:

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            # scores = hops_generator.prior(rel)

            hop_rel_lst = hops_generator(rel)
            nb_hops = len(hop_rel_lst)

            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, entity_embeddings, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, entity_embeddings, depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, entity_embeddings, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, entity_embeddings, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2, facts)

            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                entity_embeddings: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=self.depth)
        return res_sp, res_po

    def r_forward(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                  facts: List[Tensor],
                  entity_embeddings: Tensor,
                  depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = None, None
        for d in range(depth + 1):
            scores_sp, scores_po = self.depth_r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=d)
            res_sp = scores_sp if res_sp is None else torch.max(res_sp, scores_sp)
            res_po = scores_po if res_po is None else torch.max(res_po, scores_po)
        return res_sp, res_po

    def depth_r_forward(self,
                        rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                        facts: List[Tensor],
                        entity_embeddings: Tensor,
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        if depth == 0:
            return self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

        global_scores_sp = global_scores_po = None

        mask = None
        new_hops_lst = self.hops_lst

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

        # for rule_idx, (hop_generators, is_reversed) in enumerate(self.hops_lst):
        for rule_idx, (hop_generators, is_reversed) in enumerate(new_hops_lst):
            scores_sp = scores_po = None
            hop_rel_lst = hop_generators(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                sources, scores = arg1, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:

                    if mask is not None:
                        prior = prior * mask[:, rule_idx]
                        if (prior != 0.0).sum() == 0:
                            continue

                    scores = prior

                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, entity_embeddings, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, entity_embeddings, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, entity_embeddings, depth=depth - 1)
                        else:
                            scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, entity_embeddings, depth=depth - 1)

                        nb_entities = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities)
                            scores_sp = self._tnorm(scores, scores_sp)

                            # [B, S, N]
                            scores_sp = scores_sp.view(batch_size, -1, nb_entities)
                            # [B, N]
                            scores_sp, _ = torch.max(scores_sp, dim=1)

            if arg2 is not None:
                sources, scores = arg2, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:
                    scores = prior
                # scores = hop_generators.prior(rel)

                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, entity_embeddings, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, entity_embeddings, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, entity_embeddings, depth=depth - 1)
                        else:
                            _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, entity_embeddings, depth=depth - 1)

                        nb_entities = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities)
                            scores_po = self._tnorm(scores, scores_po)

                            # [B, S, N]
                            scores_po = scores_po.view(batch_size, -1, nb_entities)
                            # [B, N]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

        return global_scores_sp, global_scores_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor,
                      arg1: Optional[Tensor],
                      arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
