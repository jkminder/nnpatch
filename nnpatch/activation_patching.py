from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch
from tqdm.auto import tqdm
from loguru import logger

from nnsight import NNsight

from .api.model_api import ModelAPI
from .site import Site, Sites

def clean_run(nnmodel, sites_dict, clean_tokens, corrupted_tokens, source_attention_mask, target_attention_mask, correct_index, incorrect_index, device, validate=False, scan=False, force_model_confidence=True):
    # Clean run
    batch_range = torch.arange(0, clean_tokens.shape[0], device=device) # Batch range
    with nnmodel.trace(clean_tokens, attention_mask=source_attention_mask, scan=scan, validate=validate) as invoker:

        # Get hidden states of all layers in the network.
        # We index the output at 0 because it's a tuple where the first index is the hidden state.
        # No need to call .save() as we don't need the values after the run, just within the experiment run.
        for site_name, sites_list in sites_dict.items():
            for site in sites_list:
                site.cache(nnmodel)

        # Get logits from the lm_head.
        clean_logits = nnmodel.lm_head.output.save()
        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[batch_range, -1, correct_index.unsqueeze(0)] - clean_logits[batch_range, -1, incorrect_index]
        ).cpu().save()
    
        

    # Corrupted run
    with nnmodel.trace(corrupted_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate) as invoker:
            corrupted_logits = nnmodel.lm_head.output.save()
            # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
            corrupted_logit_diff = (
                corrupted_logits[batch_range, -1, correct_index]
                - corrupted_logits[batch_range, -1, incorrect_index]
            ).cpu().save()
        

    clean_logit_diff = clean_logit_diff.value
    corrupted_logit_diff = corrupted_logit_diff.value
    
    if (clean_logit_diff < 0).any() and force_model_confidence:
        raise ValueError("The clean logit diff is not positive for all samples. The model is not confident in the correct answer. \n" + str(clean_logit_diff))
    if (corrupted_logit_diff > 0).any() and force_model_confidence:
        raise ValueError("The corrupted logit diff is not negative for all samples. The model is not confident in the inversed answer. \n" + str(corrupted_logit_diff))
    clean_logit_diff = clean_logit_diff.mean()
    corrupted_logit_diff = corrupted_logit_diff.mean()
    logger.info(f"Clean logit diff: {clean_logit_diff}" )
    logger.info(f"Corrupted logit diff: {corrupted_logit_diff}" )
    return clean_logit_diff, corrupted_logit_diff



def patch_run(
    nnmodel: NNsight,
    sites_dict: Dict[str, List[Tuple[Tensor, Site]]],
    corrupted_tokens: Tensor,
    target_attention_mask: Tensor,
    correct_index: int,
    incorrect_index: int,
    corrupted_logit_diff,
    clean_logit_diff,
    device: str,
    validate=False,
    scan=False
):
    with torch.no_grad():
        batch_range = torch.arange(0, corrupted_tokens.shape[0], device=device) # Batch range
        for site_name, sites_list in sites_dict.items():
            for site in tqdm(sites_list, desc=str(site_name)):
                with nnmodel.trace(corrupted_tokens,attention_mask=target_attention_mask,  validate=validate, scan=scan) as invoker:
                    # Patching corrupted run at given layer and token
                    # Apply the patch from the clean hidden states to the corrupted hidden states.
                    site.patch(nnmodel)
                    patched_logits = nnmodel.lm_head.output

                    patched_logit_diff = (
                        patched_logits[batch_range, -1, correct_index]
                        - patched_logits[batch_range, -1, incorrect_index]
                    ).mean().save()

                    # Calculate the improvement in the correct token after patching.
                    patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                        clean_logit_diff - corrupted_logit_diff
                    )

                    _local = patched_result.detach().cpu().save()
                site.result = _local.value
            torch.cuda.empty_cache()
    return sites_dict

def activation_patch(
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
    results_as_tensors=True,
):
    if source_attention_mask is None:
        logger.warning("Source attention mask not provided. Using all ones.")
        source_attention_mask = torch.ones_like(source_tokens)
    if target_attention_mask is None:
        logger.warning("Target attention mask not provided. Using all ones.")
        target_attention_mask = torch.ones_like(target_tokens)
    sites_dict = sites.get_sites_dict(nnmodel, api, source_tokens)

    with torch.no_grad():
        clean_logit_diff, corrupted_logit_diff = clean_run(nnmodel, sites_dict, source_tokens, target_tokens, source_attention_mask, target_attention_mask, source_answer_index, target_answer_index, nnmodel.device, validate=validate, scan=scan, force_model_confidence=force_model_confidence)
        
        patch_run(nnmodel, sites_dict, target_tokens, target_attention_mask, source_answer_index, target_answer_index, corrupted_logit_diff, clean_logit_diff, nnmodel.device, validate=validate, scan=scan)
    
    if results_as_tensors:
        out = sites.results_to_tensor()
    else:
        out = sites.sites_dict
    return out


def activation_zero_patch(
    nnmodel: NNsight,
    api: ModelAPI,
    sites: Sites,
    target_tokens: Tensor,
    target_answer_index: Tensor,
    target_attention_mask=None,
    validate=False,
    scan=False
):
    if target_attention_mask is None:
        logger.warning("Target attention mask not provided. Using all ones.")
        target_attention_mask = torch.ones_like(target_tokens)

    sites_dict = sites.get_sites_dict(nnmodel, api, target_tokens)
    
    with torch.no_grad():
        device = nnmodel.device
        batch_range = torch.arange(0, target_tokens.shape[0], device=device) # Batch range

        logits = nnmodel.trace(target_tokens, attention_mask=target_attention_mask, trace=False).logits.detach()
        
        target_logits = logits[batch_range, -1, target_answer_index.unsqueeze(0)]
        
        for site_name, sites_list in sites_dict.items():
            for site in tqdm(sites_list, desc=str(site_name)):
                with nnmodel.trace(target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate):
                    site.patch(nnmodel, zero=True)
                    patched_logit = nnmodel.lm_head.output[batch_range, -1, target_answer_index]
                    
                    # calc the percentage change in logit
                    patched_result = (patched_logit - target_logits) / target_logits

                    _local = patched_result.detach().mean().cpu().save()
                    
                site.result = _local.value
                torch.cuda.empty_cache()


    out = sites.results_to_tensor()
    return out