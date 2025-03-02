from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch
from tqdm.auto import tqdm
from loguru import logger

from nnsight import NNsight, LanguageModel

from .api.model_api import ModelAPI
from .site import Site, Sites


class Patcher:
    def __init__(self, nnmodel: LanguageModel, api: ModelAPI, remote: bool = False):
        self.nnmodel = nnmodel
        self.remote = remote
        self.api = api

    def clean_run(
        self,
        sites_dict,
        clean_tokens,
        corrupted_tokens,
        source_attention_mask,
        target_attention_mask,
        correct_index,
        incorrect_index,
        device,
        validate=False,
        scan=False,
        force_model_confidence=True,
    ):
        raise NotImplementedError("This method should be implemented by the subclass")

    def patch_run(
        self,
        sites_dict,
        corrupted_tokens,
        target_attention_mask,
        correct_index,
        incorrect_index,
        corrupted_logit_diff,
        clean_logit_diff,
        device,
        validate=False,
        scan=False,
    ):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def activation_patch(
        self,
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
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def activation_zero_patch(
        self,
        sites: Sites,
        target_tokens: Tensor,
        target_answer_index: Tensor,
        target_attention_mask=None,
        validate=False,
        scan=False,
    ):
        raise NotImplementedError("This method should be implemented by the subclass")
    

class NNsightPatcher(Patcher):
    def __init__(self, nnmodel: NNsight, api: ModelAPI, remote: bool = False):
        super().__init__(nnmodel, api, remote)

    def clean_run(
        self,
        sites_dict: Dict[str, Dict[Site, Site]],
        clean_tokens,
        corrupted_tokens,
        source_attention_mask,
        target_attention_mask,
        correct_index,
        incorrect_index,
        device,
        validate=False,
        scan=False,
        force_model_confidence=True,
    ):
        # Clean run
        batch_range = torch.arange(0, clean_tokens.shape[0], device=device)  # Batch range
        with self.nnmodel.trace(
            clean_tokens, attention_mask=source_attention_mask, scan=scan, validate=validate, remote=self.remote
        ) as invoker:

            # Get hidden states of all layers in the network.
            # We index the output at 0 because it's a tuple where the first index is the hidden state.
            # No need to call .save() as we don't need the values after the run, just within the experiment run.
            for site_name, sites_list in sites_dict.items():
                for target_site, src_site in sites_list.items():
                    if src_site is not None:
                        src_site.cache(self.nnmodel)
                    else:
                        target_site.cache(self.nnmodel)

            # Get logits from the lm_head.
            clean_logits = self.nnmodel.lm_head.output.save()
            # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
            clean_logit_diff = (
                (
                    clean_logits[batch_range, -1, correct_index.unsqueeze(0)]
                    - clean_logits[batch_range, -1, incorrect_index]
                )
                .cpu()
                .save()
            )

        # Corrupted run
        with self.nnmodel.trace(
            corrupted_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate, remote=self.remote
        ) as invoker:
            corrupted_logits = self.nnmodel.lm_head.output.save()
            # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
            corrupted_logit_diff = (
                (corrupted_logits[batch_range, -1, correct_index] - corrupted_logits[batch_range, -1, incorrect_index])
                .cpu()
                .save()
            )

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
        return clean_logit_diff, corrupted_logit_diff

    def patch_run(
        self,
        sites_dict: Dict[str, Dict[Site, Site]],
        corrupted_tokens: Tensor,
        target_attention_mask: Tensor,
        correct_index: int,
        incorrect_index: int,
        corrupted_logit_diff,
        clean_logit_diff,
        device: str,
        validate=False,
        scan=False,
    ):
        """
        Create a patched run at each site in the sites_dict.
        
        Args:
            sites_dict: Dict[str, List[Dict[Site, Site]]]
                A dictionary of sites to patch. The keys are the site names (e.g. "resid", "o"), and the values are list of dict of sites to try patching at.
                Each dict maps from {target site: src site}.
            target_attention_mask: Tensor
                The attention mask of the target tokens.
            correct_index: int
                The index of the correct answer.
            incorrect_index: int
                The index of the incorrect answer.
                
        """

        with torch.no_grad():
            batch_range = torch.arange(0, corrupted_tokens.shape[0], device=device)  # Batch range
            for site_name, sites_list in sites_dict.items():
                for target_site, src_site in tqdm(sites_list.items(), desc=str(site_name)):
                    with self.nnmodel.trace(
                        corrupted_tokens,
                        attention_mask=target_attention_mask,
                        validate=validate,
                        scan=scan,
                        remote=self.remote,
                    ) as invoker:
                        # Patching corrupted run at given layer and token
                        # Apply the patch from the clean hidden states to the corrupted hidden states.
                        target_site.patch(self.nnmodel, src_site=src_site)
                        patched_logits = self.nnmodel.lm_head.output

                        patched_logit_diff = (
                            (
                                patched_logits[batch_range, -1, correct_index]
                                - patched_logits[batch_range, -1, incorrect_index]
                            )
                            .mean()
                            .save()
                        )

                        # Calculate the improvement in the correct token after patching.
                        patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                            clean_logit_diff - corrupted_logit_diff
                        ) 
                        # clean_logit_diff=how much the model prefers the fb answer in the fb. (should be positive)
                        # corrupted_logit_diff=how much the model prefers fb answer in the tb. (should be negative)
                        # patched_logit_diff=how much the model prefers fb answer in the tb after patching. (should be positive)
                        # clean_logit_diff - corrupted_logit_diff = how much the model prefers the fb answer in the fb case over the tb case.
                        # patched_logit_diff - corrupted_logit_diff = how much the model prefers the fb answer in the (tb case after patching) over the tb case.
                        # patched_result= what percent of the improvement in the fb answer in the tb case after patching. # we want this to be positive.

                        _local = patched_result.detach().cpu().save()
                    target_site.result = _local
                torch.cuda.empty_cache()
        return sites_dict

    def activation_patch(
        self,
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
        sites_dict = sites.get_sites_dict(self.nnmodel, self.api, source_tokens)

        with torch.no_grad():
            clean_logit_diff, corrupted_logit_diff = self.clean_run(
                sites_dict,
                source_tokens,
                target_tokens,
                source_attention_mask,
                target_attention_mask,
                source_answer_index,
                target_answer_index,
                self.nnmodel.device if not self.remote else None,
                validate=validate,
                scan=scan,
                force_model_confidence=force_model_confidence,
            )

            self.patch_run(
                sites_dict,
                target_tokens,
                target_attention_mask,
                source_answer_index,
                target_answer_index,
                corrupted_logit_diff,
                clean_logit_diff,
                self.nnmodel.device if not self.remote else None,
                validate=validate,
                scan=scan,
            )

        if results_as_tensors:
            out = sites.results_to_tensor()
        else:
            out = sites.sites_dict
        return out

    def activation_zero_patch(
        self,
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

        sites_dict = sites.get_sites_dict(self.nnmodel, self.api, target_tokens)

        with torch.no_grad():
            device = self.nnmodel.device
            batch_range = torch.arange(0, target_tokens.shape[0], device=device)  # Batch range

            logits = self.nnmodel.trace(target_tokens, attention_mask=target_attention_mask, trace=False).logits.detach()

            target_logits = logits[batch_range, -1, target_answer_index.unsqueeze(0)]

            for site_name, sites_list in sites_dict.items():
                for site in tqdm(sites_list, desc=str(site_name)):
                    with self.nnmodel.trace(
                        target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate, remote=self.remote
                    ):
                        site.patch(self.nnmodel, zero=True)
                        patched_logit = self.nnmodel.lm_head.output[batch_range, -1, target_answer_index]

                        # calc the percentage change in logit
                        patched_result = (patched_logit - target_logits) / target_logits

                        _local = patched_result.detach().mean().cpu().save()

                    site.result = _local
                    torch.cuda.empty_cache()

        out = sites.results_to_tensor()
        return out

class LanguageModelPatcher(Patcher):
    def __init__(self, nnmodel: LanguageModel, api: ModelAPI, remote: bool = False):
        super().__init__(nnmodel, api, remote)

    def clean_run(
        self,
        sites_dict,
        clean_tokens,
        corrupted_tokens,
        source_attention_mask,
        target_attention_mask,
        correct_index,
        incorrect_index,
        device,
        validate=False,
        scan=False,
        force_model_confidence=True,
    ):
        if isinstance(self.nnmodel, LanguageModel):
            if source_attention_mask is not None:
                raise ValueError("Source attention mask not supported for LanguageModel (it is computed automatically)")
            if target_attention_mask is not None:
                raise ValueError("Target attention mask not supported for LanguageModel (it is computed automatically)")

        # Clean run
        batch_range = torch.arange(0, clean_tokens.shape[0], device=device)
        with self.nnmodel.trace(
            clean_tokens, scan=scan, validate=validate, remote=self.remote
        ) as invoker:
            # Get hidden states of all layers in the network.
            # We index the output at 0 because it's a tuple where the first index is the hidden state.
            # No need to call .save() as we don't need the values after the run, just within the experiment run.
            for site_name, sites_list in sites_dict.items():
                for target_site, src_site in sites_list.items():
                    if src_site is not None:
                        src_site.cache(self.nnmodel)
                    else:
                        target_site.cache(self.nnmodel)

            # Get logits from the lm_head.
            clean_logits = self.nnmodel.lm_head.output.save()
            # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
            clean_logit_diff = (
                (
                    clean_logits[batch_range, -1, correct_index.unsqueeze(0)]
                    - clean_logits[batch_range, -1, incorrect_index.unsqueeze(0)]
                )
                .cpu()
                .save()
            )

        # Corrupted run
        with self.nnmodel.trace(
            corrupted_tokens, scan=scan, validate=validate, remote=self.remote
        ) as invoker:
            batch_range = torch.arange(0, clean_tokens.shape[0], device=device)
            corrupted_logits = self.nnmodel.lm_head.output.save()
            # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
            corrupted_logit_diff = (
                (corrupted_logits[batch_range, -1, correct_index.unsqueeze(0)] - corrupted_logits[batch_range, -1, incorrect_index.unsqueeze(0)])
                .cpu()
                .save()
            )

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
        return clean_logit_diff, corrupted_logit_diff

    def patch_run(
        self,
        sites_dict: Dict[str, List[Tuple[Tensor, Site]]],
        corrupted_tokens: Tensor,
        target_attention_mask: Tensor,
        correct_index: int,
        incorrect_index: int,
        corrupted_logit_diff,
        clean_logit_diff,
        device: str,
        validate=False,
        scan=False,
    ):
        with torch.no_grad():
            with self.nnmodel.trace(remote=self.remote, validate=validate, scan=scan) as tracer:
                for site_name, sites_list in sites_dict.items():
                    for target_site, src_site in tqdm(sites_list.items(), desc=str(site_name)):
                        with tracer.invoke(
                            corrupted_tokens,
                        ) as invoker:
                            batch_range = torch.arange(0, corrupted_tokens.shape[0], device=device)  # Batch range
                            # Patching corrupted run at given layer and token
                            # Apply the patch from the clean hidden states to the corrupted hidden states.
                            target_site.patch(self.nnmodel, src_site=src_site)
                            patched_logits = self.nnmodel.lm_head.output

                            patched_logit_diff = (
                                (
                                    patched_logits[batch_range, -1, correct_index.unsqueeze(0)]
                                    - patched_logits[batch_range, -1, incorrect_index.unsqueeze(0)]
                                )
                                .mean()
                                .save()
                            )

                            # Calculate the improvement in the correct token after patching.
                            patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                                clean_logit_diff - corrupted_logit_diff
                            )

                            _local = patched_result.detach().cpu().save()
                        target_site.result = _local
                        #{"patching_result": _local, "patched_logit_diff": patched_logit_diff, "corrupted_logit_diff": corrupted_logit_diff, "clean_logit_diff": clean_logit_diff}
                    torch.cuda.empty_cache()
        return sites_dict

    def activation_patch(
        self,
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
        sites_dict = sites.get_sites_dict(self.nnmodel, self.api, source_tokens)

        with torch.no_grad():
            clean_logit_diff, corrupted_logit_diff = self.clean_run(
                sites_dict=sites_dict,
                clean_tokens=source_tokens,
                corrupted_tokens=target_tokens,
                source_attention_mask=source_attention_mask,
                target_attention_mask=target_attention_mask,
                correct_index=source_answer_index,
                incorrect_index=target_answer_index,
                device=self.nnmodel.device if not self.remote else None,
                validate=validate,
                scan=scan,
                force_model_confidence=force_model_confidence,
            )

            self.patch_run(
                sites_dict=sites_dict,
                corrupted_tokens=target_tokens,
                target_attention_mask=target_attention_mask,
                correct_index=source_answer_index,
                incorrect_index=target_answer_index,
                corrupted_logit_diff=corrupted_logit_diff,
                clean_logit_diff=clean_logit_diff,
                device=self.nnmodel.device if not self.remote else None,
                validate=validate,
                scan=scan,
            )

        if results_as_tensors:
            out = sites.results_to_tensor()
        else:
            out = sites.sites_dict
        return out

    def activation_zero_patch(
        self,
        sites: Sites,
        target_tokens: Tensor,
        target_answer_index: Tensor,
        target_attention_mask=None,
        validate=False,
        scan=False,
    ):
        if target_attention_mask is not None:
            raise ValueError("Target attention mask not supported for LanguageModel (it is computed automatically)")

        sites_dict = sites.get_sites_dict(self.nnmodel, self.api, target_tokens)

        with torch.no_grad():
            device = self.nnmodel.device
            batch_range = torch.arange(0, target_tokens.shape[0], device=device)  # Batch range

            logits = self.nnmodel.trace(target_tokens, attention_mask=target_attention_mask, trace=False, remote=self.remote).logits.detach()

            target_logits = logits[batch_range, -1, target_answer_index.unsqueeze(0)]

            for site_name, sites_list in sites_dict.items():
                for site in tqdm(sites_list, desc=str(site_name)):
                    with self.nnmodel.trace(
                        target_tokens, attention_mask=target_attention_mask, scan=scan, validate=validate, remote=self.remote
                    ):
                        site.patch(self.nnmodel, zero=True)
                        patched_logit = self.nnmodel.lm_head.output[batch_range, -1, target_answer_index]

                        # calc the percentage change in logit
                        patched_result = (patched_logit - target_logits) / target_logits

                        _local = patched_result.detach().mean().cpu().save()

                    site.result = _local
                    torch.cuda.empty_cache()

        out = sites.results_to_tensor()
        return out
