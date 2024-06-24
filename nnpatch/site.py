from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch

from nnsight.model import NNsight

from .api.model_api import ModelAPI
from .utils import hidden_to_head, head_to_hidden

class Site:
    def __init__(
        self, 
        api: ModelAPI,
        node_name, 
        layer: Optional[int] = None,
        head: Optional[int] = None,
        seq_pos: "SeqPos" = None,
    ):
        self.api = api
        if layer is not None: assert layer >= 0, "Doesn't accept negative layers."

        self.component_name = node_name
        self.layer = layer

        if head is not None: assert isinstance(head, int), f"head should be an int, not {type(head)}"
        self.head = head
        self.seq_pos = seq_pos
        
        self._result = None

    @abstractmethod
    def cache(self, nnmodel):
        pass
    
    @abstractmethod
    def patch(self, nnmodel):
        pass

    @staticmethod
    def get_site(api, node_name, layer, head, seq_pos):
        if node_name in ["q", "k", "v", "o"]:
            return HeadSite(api, node_name, layer, head, seq_pos)
        if node_name == "mlp":
            return MLPSite(api, layer, seq_pos)
        if node_name == "resid":
            return ResidSite(api, layer, seq_pos)
        if node_name == "attn":
            return AttnSite(api, layer, seq_pos)
        raise ValueError(f"Unknown node name: {node_name}")
    
    @property
    def result(self):
        return self._result
    
    @result.setter
    def result(self, value):    
        self._result = value
    
    def __repr__(self) -> str:
        return f"{self.component_name}({self.layer}, {self.head}, {self.seq_pos})"
    
class HeadSite(Site):
    _cache = defaultdict(dict)
    def __init__(self, api, head_type, layer, head, seq_pos=None):
        assert head is not None, "Head must be specified for Head node."
        super().__init__(api, head_type, layer, head, seq_pos)
    
    def cache(self, nnmodel):
        if self.layer not in HeadSite._cache or self.component_name not in HeadSite._cache[self.layer]:
            HeadSite._cache[self.layer][self.component_name] = getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output.detach().cpu().save()
    
    def num_heads(self, model):
        return self.api.N_QO_HEADS(model) if self.component_name in ["q", "o"] else self.api.N_KV_HEADS(model)
    
    def patch(self, nnmodel):
        clean = HeadSite._cache[self.layer][self.component_name]
        n_heads = self.num_heads(nnmodel)
        dirty = getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output
        dirty = hidden_to_head(dirty, n_heads)
        clean = hidden_to_head(clean, n_heads) # batch pos head_index d_head
        dirty[:, self.seq_pos, self.head] = clean[:, self.seq_pos, self.head]
        getattr(self.api, f"get_{self.component_name}")(nnmodel, self.layer).output = head_to_hidden(dirty)
        
class MLPSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "mlp", layer, None, seq_pos)
    
    def cache(self, nnmodel):
        self._cache = self.api.get_mlp(nnmodel, self.layer).output[:, self.seq_pos, :].detach().cpu().save()
        
    def patch(self, nnmodel):
        self.api.get_mlp(nnmodel, self.layer).output[:, self.seq_pos, :] = self._cache

                
class ResidSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "resid", layer, None, seq_pos)
    
    def cache(self, nnmodel):
        self._cache = self.api.get_layer(nnmodel, self.layer).output[0][:, self.seq_pos, :].detach().cpu().save()
    
    def patch(self, nnmodel):
        dirty = self.api.get_layer(nnmodel, self.layer)
        dirty_output = dirty.output[0]
        dirty_output[:, self.seq_pos, :] = self._cache
        dirty.output = (dirty_output, dirty.output[1])
        
class AttnSite(Site):
    def __init__(self, api, layer, seq_pos=None):
        super().__init__(api, "attn", layer, None, seq_pos)
        
    def cache(self, nnmodel):
        self._cache = self.api.get_attn(nnmodel, self.layer).output[0][:, self.seq_pos, :].detach().cpu().save()
        
    def patch(self, nnmodel):
        dirty = self.api.get_attn(nnmodel, self.layer)
        dirty_output = dirty.output[0]
        dirty_output[:, self.seq_pos, :] = self._cache
        dirty.output = (dirty_output, dirty.output[1], dirty.output[2])


class Sites:
    def __init__(
        self, 
        site_names: Union[str, List[str]], 
        seq_pos_type: str = "custom",
        seq_pos: Tensor = None,
        layers: List[int] = None
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

    def iter(
        self,
        shape_names: List[str],
        shape_values: Dict[str, int],
        seq_pos_indices: Optional[Tensor]
    ):
        layers = range(shape_values["layer"]) if self.layers is None else self.layers
        heads = [None]
        seq_pos = seq_pos_indices
        
        if "head_qo" in shape_names: heads = range(shape_values["head_qo"])
        elif "head_kv" in shape_names: heads = range(shape_values["head_kv"])
        for layer in layers:
            for head in heads:
                for pos in seq_pos:
                    yield (layer, head, pos)
            
        
    def get_sites_dict(
        self, 
        model: NNsight,
        api: ModelAPI,
        tensor = None
    ) -> Dict[str, List[Tuple[Tensor, Site]]]:
        
        # Get a dict we can use to convert from shape names into actual shape values
        # We filter dict down, because we might not need all of these (e.g. we'll never need neurons and heads at the same time!)
        batch_size, seq_len = tensor.shape[:2]
        seq_len = seq_len # for lastk
        self.shape_values_all = {"seq_pos": seq_len, "layer": api.N_LAYERS(model), "head_qo": api.N_QO_HEADS(model), "head_kv": api.N_KV_HEADS(model)}


        self.sites_dict = {}
        self.shape_values = {}

        # If iterating, get a list of sequence positions we'll be iterating over
        if self.seq_pos_type == "each":
            seq_pos_indices = torch.arange(seq_len)
        elif self.seq_pos_type is None:
            seq_pos_indices = [torch.arange(seq_len)]
        elif self.seq_pos_type == "last":
            seq_pos_indices = [torch.tensor([seq_len - 1])]
        elif self.seq_pos_type == "lastk":
            seq_pos_indices = torch.arange(seq_len - self.seq_pos, seq_len)
        elif self.seq_pos_type == "custom_constant":
            seq_pos_indices = [self.seq_pos]
        elif self.seq_pos_type == "custom":
            seq_pos_indices = self.seq_pos
        else:
            raise ValueError(f"Unknown seq_pos_type: {self.seq_pos_type}")
            
        for site_name, shape_names in self.shape_names.items():
            self.sites_dict[site_name] = [
                Site.get_site(api, site_name, layer, head, seq_pos)
                for layer, head, seq_pos in self.iter(shape_names, self.shape_values_all, seq_pos_indices)
            ]
        return self.sites_dict
    
    def results_to_tensor(self):
        out = {}
        for site_name, sites in self.sites_dict.items():
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
                out[site_name][*index] = site.result.float()
                        
        return out