from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch
from copy import deepcopy

from nnsight import NNsight

from .api.model_api import ModelAPI
from .utils import hidden_to_head, head_to_hidden
    
class Site:
    def __init__(
        self, 
        api: ModelAPI,
        site_name, 
        layer: Optional[int] = None,
        head: Optional[int] = None,
        seq_pos: "SeqPos" = None,
    ):
        self.api = api
        if layer is not None: assert layer >= 0, "Doesn't accept negative layers."

        self.component_name = site_name
        self.layer = layer

        if head is not None: assert isinstance(head, int), f"head should be an int, not {type(head)}"
        self.head = head
        self.seq_pos = seq_pos
        
        self._result = None
        self._cache = None
        self._gradient_cache = None

    @abstractmethod
    def cache(self, nnmodel, gradient=False):
        pass
        
    @abstractmethod
    def patch(self, nnmodel, zero=False):
        pass

    @staticmethod
    def get_site(api, site_name, layer, head, seq_pos, cache_name="default"):
        if site_name in ["q", "k", "v", "o"]:
            return HeadSite(api, site_name, layer, head, seq_pos, cache_name)
        if site_name == "mlp":
            return MLPSite(api, layer, seq_pos)
        if site_name == "resid":
            return ResidSite(api, layer, seq_pos)
        if site_name == "attn":
            return AttnSite(api, layer, seq_pos)
        raise ValueError(f"Unknown node name: {site_name}")
    
    @property
    def result(self):
        return self._result
    
    @result.setter
    def result(self, value):    
        self._result = value
        
    def reset(self):
        self._cache = None
        self._gradient_cache = None
        self._result = None
    
    def __repr__(self) -> str:
        return f"{self.component_name}({self.layer}, {self.head}, {self.seq_pos})"
    
    def _attribution_args_validation(self, nnmodel, clean_site, zero):
        if clean_site is None and not zero:
            raise ValueError("Clean site must be specified for attribution (or use zero attribution).")
        if self._cache is None or (clean_site is not None and clean_site._cache is None) or self._gradient_cache is None:
            raise ValueError("Cache not found. Please run .cache() (with gradient=True on corrupted) before running attribution.")
        
    def attribution(self, nnmodel, clean_site=None, zero=False):
        self._attribution_args_validation(nnmodel, clean_site, zero)        
        if zero:
            attribution = -self._cache * self._gradient_cache
        else:
            attribution = (clean_site._cache - self._cache) * self._gradient_cache
        self.result = attribution.sum()
        return self.result
    
    def average_cache_over_samples(self, nnmodel, other_sites=[]):
        if self._cache is None:
            raise ValueError("Cache not found. Please run .cache() before running average_cache_over_samples.")
        mean = self._cache
        for site in other_sites:
            mean = torch.cat([mean, site._cache], dim=0)
        mean = mean.mean(dim=0)
        self._cache = mean
        if self._gradient_cache is not None:
            mean = self._gradient_cache
            for site in other_sites:
                mean = torch.cat([mean, site._gradient_cache], dim=0)
            mean = mean.mean(dim=0)
            self._gradient_cache = mean

class HeadSite(Site):
    _cache = defaultdict(lambda: defaultdict(dict))
    _gradient_cache = defaultdict(lambda: defaultdict(dict))
    def __init__(self, api, head_type, layer, head, seq_pos, cache_name):
        assert head is not None, "Head must be specified for Head node."
        super().__init__(api, head_type, layer, head, seq_pos)
        self.cache_name = cache_name
        
    @staticmethod
    def reset():
        HeadSite._cache = defaultdict(lambda: defaultdict(dict))
        HeadSite._gradient_cache = defaultdict(lambda: defaultdict(dict))
    
    def reset(self):
        HeadSite._cache[self.cache_name] = defaultdict(dict)
        HeadSite._gradient_cache[self.cache_name] = defaultdict(dict)
        
    def cache(self, nnmodel, gradient=False):
        if self.component_name not in HeadSite._cache[self.cache_name][self.layer]:
            proxy = getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output
            HeadSite._cache[self.cache_name][self.layer][self.component_name] = proxy.detach().cpu().save()

            if gradient and (self.component_name not in HeadSite._gradient_cache[self.cache_name][self.layer]):
                HeadSite._gradient_cache[self.cache_name][self.layer][self.component_name] = proxy.grad.detach().cpu().save()
            
    def num_heads(self, model):
        return self.api.N_QO_HEADS(model) if self.component_name in ["q", "o"] else self.api.N_KV_HEADS(model)
    
    def patch(self, nnmodel, zero=False):
        n_heads = self.num_heads(nnmodel)
        if zero:
            cache = 0
        else:
            clean = self.get_cache()
            clean = hidden_to_head(clean, n_heads) # batch pos head_index d_head
            cache = clean[:, self.seq_pos, self.head]
        dirty = getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output
        dirty = hidden_to_head(dirty, n_heads)
        dirty[:, self.seq_pos, self.head] = cache
        getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output = head_to_hidden(dirty)
        
    def get_cache(self):
        return HeadSite._cache[self.cache_name][self.layer][self.component_name]
    
    def get_gradient_cache(self):
        return HeadSite._gradient_cache[self.cache_name][self.layer][self.component_name]
    
    def attribution(self, nnmodel, clean_site=None, zero=False):
        self._attribution_args_validation(nnmodel, clean_site, zero)
        n_heads = self.num_heads(nnmodel)

        corrupted = hidden_to_head(self.get_cache(), n_heads)[:, self.seq_pos, self.head]
        gradient = hidden_to_head(self.get_gradient_cache(), n_heads)[:, self.seq_pos, self.head]
        if zero:
            attribution = -corrupted * gradient
        else:
            clean = hidden_to_head(clean_site.get_cache(), n_heads)[: , self.seq_pos, self.head]
            attribution = (clean - corrupted) * gradient
        self.result = attribution.sum(dim=(1,2)).mean()
        return self.result
    
    def average_cache_over_samples(self, nnmodel, other_sites=[]):
        n_heads = self.num_heads(nnmodel)
        cache = self.get_cache()
        cache = hidden_to_head(cache, n_heads) # batch pos head_index d_head
        mean = cache[:, self.seq_pos, self.head]
        for site in other_sites:
            assert type(site) == HeadSite, "Only HeadSite is supported for averaging cache over samples."
            other_cache = hidden_to_head(site.get_cache(), n_heads)
            other_cache = other_cache[:, self.seq_pos, site.head]
            mean = torch.cat([mean, other_cache], dim=0)
        mean = mean.mean(dim=0)
            
        cache[:, self.seq_pos, self.head] = mean
        HeadSite._cache[self.cache_name][self.layer][self.component_name] = head_to_hidden(cache)
        
        if self.component_name in HeadSite._gradient_cache[self.cache_name][self.layer]:
            gradient_cache = self.get_gradient_cache()
            gradient_cache = hidden_to_head(gradient_cache, n_heads)
            mean = gradient_cache[:, self.seq_pos, self.head]
            for site in other_sites:
                other_cache = hidden_to_head(site.get_gradient_cache(), n_heads)
                other_cache = other_cache[:, self.seq_pos, site.head]
                mean = torch.cat([mean, other_cache], dim=0)
            gradient_cache[:, self.seq_pos, self.head] = mean
            HeadSite._gradient_cache[self.cache_name][self.layer][self.component_name] = head_to_hidden(gradient_cache)

class MLPSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "mlp", layer, None, seq_pos)
    
    def cache(self, nnmodel, gradient=False):
        proxy = self.api.get_mlp(nnmodel, self.layer).output
        self._cache = proxy[:, self.seq_pos, :].detach().cpu().save()
        if gradient:
            self._gradient_cache = proxy.grad[:, self.seq_pos, :].detach().cpu().save()
            
    def patch(self, nnmodel, zero=False):
        if zero:
            self.api.get_mlp(nnmodel, self.layer).output[:, self.seq_pos, :] = 0
        else:
            self.api.get_mlp(nnmodel, self.layer).output[:, self.seq_pos, :] = self._cache

                
class ResidSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "resid", layer, None, seq_pos)
    
    def cache(self, nnmodel, gradient=False):
        proxy = self.api.get_layer(nnmodel, self.layer).output[0]
        self._cache = proxy[:, self.seq_pos, :].detach().cpu().save()
        if gradient:
            self._gradient_cache = proxy.grad[:, self.seq_pos, :].cpu().save()

    def value(self):
        self._cache = self._cache.value
        self._gradient_cache = self._gradient_cache.value
        
    def patch(self, nnmodel, zero=False):
        
        dirty = self.api.get_layer(nnmodel, self.layer)
        dirty_output = dirty.output[0]
        if zero:
            dirty_output[:, self.seq_pos, :] = 0
        else:
            dirty_output[:, self.seq_pos, :] = self._cache
        dirty.output = (dirty_output, dirty.output[1])
        
class AttnSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "attn", layer, None, seq_pos)
        
    def cache(self, nnmodel, gradient=False):
        proxy = self.api.get_attn(nnmodel, self.layer).output[0][:, self.seq_pos, :]
        self._cache = proxy.detach().cpu().save()
        if gradient:
            self._gradient_cache = proxy.grad.detach().cpu().save()
            
    def patch(self, nnmodel, zero=False):
        dirty = self.api.get_attn(nnmodel, self.layer)
        dirty_output = dirty.output[0]
        if zero:
            dirty_output[:, self.seq_pos, :] = self._cache
        else:
            dirty_output[:, self.seq_pos, :] = 0
        dirty.output = (dirty_output, dirty.output[1], dirty.output[2])

class MultiSite(Site):
    def __init__(self, sites: List[Site]):
        layer = []
        head = []
        seq_pos = []
        self.component_names = []
        for site in sites:
            layer.append(site.layer)
            if site.head is not None:
                head.append(site.head)
            seq_pos.extend(site.seq_pos.tolist())
            self.component_names.append(site.component_name)
        
        self.component_names = tuple(set(self.component_names))
        self.component_name = f"MultiSite({self.component_names})"
        layer = torch.tensor(list(set(layer)))
        
        head = torch.tensor(list(set(head)))
        seq_pos = torch.tensor(list(set(seq_pos)))
        self.head = head
        self.layer = layer
        self.seq_pos = seq_pos
        self.result = None
        self.sites = sites
        
    def cache(self, nnmodel, gradient=False):
        for site in self.sites:
            site.cache(nnmodel, gradient)
            
    def patch(self, nnmodel, zero=False):
        for site in self.sites:
            site.patch(nnmodel, zero)
            
    def attribution(self, nnmodel, clean_site=None, zero=False):
        for site in self.sites:
            site.attribution(nnmodel, clean_site, zero)
        self.result = sum([site.result for site in self.sites])
        return self.result

    def reset(self):
        for site in self.sites:
            site.reset()
        self.result = None
        
    def average_cache_over_samples(self, nnmodel, other_sites=[]):
        for site in other_sites:
            assert isinstance(site, MultiSite), "Only MultiSite is supported for other_sites when averaging cache over samples."
            assert len(site.sites) == len(self.sites), "Other MultiSite should have the same number of sites."
        for i, site in enumerate(self.sites):
            site.average_cache_over_samples(nnmodel, [site.sites[i] for site in other_sites])
            
            
class Sites:
    def __init__(
        self, 
        site_names: Union[str, List[str]], 
        seq_pos_type: str = "custom",
        seq_pos: Tensor = None,
        layers: List[int] = None,
        sites_dict = None
    ):
        self.seq_pos = seq_pos
        self.seq_pos_type = seq_pos_type
        self.layers = layers
        # Get site_names into a list, for consistency
        self.component_names = [site_names] if isinstance(site_names, str) else site_names

        # Figure out what the dimensions of the output will be (i.e. for our path patching iteration)
        self.shape_names = {}
        for site in self.component_names:
            # Everything has a "layer" dim
            self.shape_names[site] = ["layer"]
            # Add "seq_pos", if required
            if seq_pos_type in ["each", "lastk", "custom"]: self.shape_names[site].append("seq_pos")
            # Add "head" and "neuron", if required
            if site in ["q",  "o"]: self.shape_names[site].append("head_qo")
            if site in ["k", "v"]: self.shape_names[site].append("head_kv")

        # Result:
        #   self.shape_names = {"z": ["layer", "seq_pos", "head"], ...}
        # where each value is the list of things we'll be iterating over.

        self.range_config = []
        self._sites_dict = sites_dict
        
    @staticmethod
    def from_list_of_dicts(dict_list):
        site_names = []
        for d in dict_list:
            site_names.append(d["site_name"])

        site_names = list(set(site_names))
        sites = Sites(site_names, seq_pos_type="custom")            

        config_range = {site_name : [] for site_name in site_names}
        for d in dict_list:
            config_range[d["site_name"]].append((d["layer"], d["head"], d["seq_pos"]))
        sites.range_config = config_range
        return sites

    @staticmethod
    def from_list_of_sites(sites_list):
        # Requires special handling for MultiSite because they have a tuple as component name
        component_names = []
        for site in sites_list:
            if isinstance(site, MultiSite):
                component_names.extend(site.component_names)
            else:
                component_names.append(site.component_name)
        sites_dict = {site.component_name: [] for site in sites_list}
        for site in sites_list:
            sites_dict[site.component_name].append(site)
        sites = Sites(component_names, seq_pos_type="custom", sites_dict=sites_dict)
        return sites
    
    def iter(
        self,
        site_name: str,
        shape_names: List[str],
        shape_values: Dict[str, int],
        seq_pos_indices: Optional[Tensor]
    ):
        layers = range(shape_values["layer"]) if self.layers is None else self.layers
        heads = [None]
        seq_pos = seq_pos_indices
        
        if not len(self.range_config):
            if "head_qo" in shape_names: heads = range(shape_values["head_qo"])
            elif "head_kv" in shape_names: heads = range(shape_values["head_kv"])
            for layer in layers:
                for head in heads:
                    for pos in seq_pos:
                        yield (layer, head, pos)
        else:
            for layer, head, pos in self.range_config[site_name]:
                yield (layer, head, pos)
        
    def get_sites_dict(
        self, 
        model: NNsight,
        api: ModelAPI,
        tensor = None,
        cache_name = "default"
    ) -> Dict[str, List[Tuple[Tensor, Site]]]:
        # Get a dict we can use to convert from shape names into actual shape values

        if tensor is not None:
            seq_len = tensor.shape[1]
        else:
            assert self.seq_pos_type not in ["each", "lastk", "custom"], "seq_pos_type requires tokens to be specified."
            seq_len = 1
        self.shape_values_all = {"seq_pos": seq_len, "layer": api.N_LAYERS(model), "head_qo": api.N_QO_HEADS(model), "head_kv": api.N_KV_HEADS(model)}

        if self._sites_dict is not None:
            self.sites_dict = deepcopy(self._sites_dict)
            return self.sites_dict

        self.sites_dict = {}
        self.shape_values = {}

        # If iterating, get a list of sequence positions we'll be iterating over
        if self.seq_pos_type == "each":
            seq_pos_indices = torch.arange(seq_len)
        elif self.seq_pos_type is None:
            seq_pos_indices = [torch.arange(seq_len)]
        elif self.seq_pos_type == "last":
            seq_pos_indices = [torch.tensor([-1])]
        elif self.seq_pos_type == "lastk":
            seq_pos_indices = torch.arange(-self.seq_pos, 0)
        elif self.seq_pos_type == "custom_constant":
            seq_pos_indices = [self.seq_pos]
        elif self.seq_pos_type == "custom":
            seq_pos_indices = self.seq_pos
        else:
            raise ValueError(f"Unknown seq_pos_type: {self.seq_pos_type}")
            
        for site_name, shape_names in self.shape_names.items():
            self.sites_dict[site_name] = [
                Site.get_site(api, site_name, layer, head, seq_pos, cache_name)
                for layer, head, seq_pos in self.iter(site_name, shape_names, self.shape_values_all, seq_pos_indices)
            ]

        return self.sites_dict

    def results_to_tensor(self):
        out = {}
        for site_name, sites in self.sites_dict.items():
            if site_name.startswith("MultiSite"):
                out[site_name] = torch.tensor([site.result for site in sites])
                print("AL", out[site_name])
                continue
            out[site_name] = torch.zeros(*[self.shape_values_all[key] for key in self.shape_names[site_name]])
            for site in sites:
                index = []
                for key in self.shape_names[site_name]:
                    if key == "layer":
                        index.append(site.layer)
                    if key in ["head_qo", "head_kv"]:
                        index.append(site.head)
                    if key == "seq_pos":
                        index.append(torch.tensor(site.seq_pos))
                out[site_name][*index] = site.result.cpu().detach().float()

        return out


def batched_average_cache(nnmodel, tokens, attention_mask, site, batch_size, gradient=False):  
    # Split the tokens and attention mask into batches
    tokens = tokens.split(batch_size)
    attention_mask = attention_mask.split(batch_size)
    
    n_batches = len(tokens)
    sites = [deepcopy(site) for i in range(n_batches)]
    for i, site in enumerate(sites[1:], start=1):
        if isinstance(site, HeadSite):
            site.cache_name = f"{site.cache_name}_batch{i}"

    for i, (tokens_batch, attention_mask_batch, site) in enumerate(zip(tokens, attention_mask, sites)):
        with nnmodel.trace(tokens_batch, attention_mask=attention_mask_batch):
            site.cache(nnmodel, gradient=gradient)
            
    site = sites[0]
    site.average_cache_over_samples(nnmodel, other_sites=sites[1:])
    return site