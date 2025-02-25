from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch
from tqdm.auto import tqdm
from loguru import logger
from copy import deepcopy
from nnsight import NNsight

from .api.model_api import ModelAPI
from .site import Site, Sites

import time


def attribution_patch(
    nnmodel: NNsight,
    api: ModelAPI,
    sites: Sites,
    source_tokens: Tensor,
    target_tokens: Tensor,
    source_answer_index: Tensor,
    target_answer_index: Tensor,
    source_attention_mask=None,
    target_attention_mask=None,
    validate=False,
    scan=False,
    force_model_confidence=True,
):
    if source_attention_mask is None:
        logger.warning("Source attention mask not provided. Using all ones.")
        source_attention_mask = torch.ones_like(source_tokens)
    if target_attention_mask is None:
        logger.warning("Target attention mask not provided. Using all ones.")
        target_attention_mask = torch.ones_like(target_tokens)

    clean_sites_dict = sites.get_sites_dict(nnmodel, api, source_tokens, cache_name="clean")
    corrupted_sites = deepcopy(sites)
    corrupted_sites_dict = corrupted_sites.get_sites_dict(nnmodel, api, source_tokens, cache_name="corrupted")

    device = nnmodel.device

    batch_range = torch.arange(0, source_tokens.shape[0], device=device)  # Batch range

    corrupted_logits = nnmodel.trace(target_tokens, attention_mask=target_attention_mask, trace=False).logits.detach()

    corrupted_logit_diff = (
        corrupted_logits[batch_range, -1, source_answer_index.unsqueeze(0)]
        - corrupted_logits[batch_range, -1, target_answer_index]
    )

    with nnmodel.trace(source_tokens, attention_mask=source_attention_mask, scan=scan, validate=validate):
        for site_name, sites_list in clean_sites_dict.items():
            for site in sites_list:
                site.cache(nnmodel)

        # Get logits from the lm_head.
        clean_logits = nnmodel.lm_head.output.save()

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            (
                clean_logits[batch_range, -1, source_answer_index.unsqueeze(0)]
                - clean_logits[batch_range, -1, target_answer_index]
            )
            .detach()
            .save()
        )

    clean_logit_diff = clean_logit_diff.value

    if (clean_logit_diff < 0).any() and force_model_confidence:
        raise ValueError(
            "The clean logit diff is not positive for all samples. The model is not confident in the correct answer. \n"
            + str(clean_logit_diff)
        )
    if (corrupted_logit_diff > 0).any() and force_model_confidence:
        raise ValueError(
            "The corrupted logit diff is not negative for all samples. The model is not confident in the inversed answer. \n"
            + str(corrupted_logit_diff)
        )
    clean_logit_diff = clean_logit_diff.mean()
    corrupted_logit_diff = corrupted_logit_diff.mean()
    logger.info(f"Clean logit diff: {clean_logit_diff}")
    logger.info(f"Corrupted logit diff: {corrupted_logit_diff}")

    def error_function(logit_diff):
        # Calculate the improvement in the correct token after patching.
        return (logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    with nnmodel.trace(target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate):
        for site_name, sites_list in corrupted_sites_dict.items():
            for site in sites_list:
                site.cache(nnmodel, gradient=True)

        logits = nnmodel.lm_head.output.cpu().save()
        logit_diff = logits[batch_range, -1, source_answer_index] - logits[batch_range, -1, target_answer_index]

        loss = error_function(logit_diff).sum()
        loss.backward()

    for site_name, sites_list in corrupted_sites_dict.items():
        for i, site in enumerate(sites_list):
            site.attribution(nnmodel, clean_sites_dict[site_name][i])

    out = corrupted_sites.results_to_tensor()
    return out


def attribution_zero_patch(
    nnmodel: NNsight,
    api: ModelAPI,
    sites: Sites,
    target_tokens: Tensor,
    target_answer_index: Tensor,
    target_attention_mask=None,
    validate=False,
    scan=False,
):
    if target_attention_mask is None:
        logger.warning("Target attention mask not provided. Using all ones.")
        target_attention_mask = torch.ones_like(target_tokens)

    sites_dict = sites.get_sites_dict(nnmodel, api, target_tokens)

    device = nnmodel.device
    batch_range = torch.arange(0, target_tokens.shape[0], device=device)  # Batch range

    with nnmodel.trace(target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate):
        for site_name, sites_list in sites_dict.items():
            for site in sites_list:
                site.cache(nnmodel, gradient=True)

        logits = nnmodel.lm_head.output.cpu().save()
        target_logits = logits[batch_range, -1, target_answer_index.unsqueeze(0)]
        target_logits_detached = target_logits.detach()
        loss = (target_logits - target_logits_detached / target_logits_detached).sum()
        loss.backward()

    for site_name, sites_list in sites_dict.items():
        for i, site in enumerate(sites_list):
            site.attribution(nnmodel, zero=True)

    out = sites.results_to_tensor()
    return out
