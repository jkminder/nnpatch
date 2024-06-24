from .model_api import ModelAPI

class Llama3(ModelAPI):
    @staticmethod
    def get_q(model, layer_idx):
        return model.model.layers[layer_idx].self_attn.q_proj
    
    @staticmethod
    def get_k(model, layer_idx):
        return model.model.layers[layer_idx].self_attn.k_proj
    
    @staticmethod
    def get_v(model, layer_idx):
        return model.model.layers[layer_idx].self_attn.v_proj
    
    @staticmethod
    def get_o(model, layer_idx):
        return model.model.layers[layer_idx].self_attn.o_proj
    
    @staticmethod
    def get_layer(model, layer_idx):
        return model.model.layers[layer_idx]
    
    @staticmethod
    def get_attn(model, layer_idx):
        return model.model.layers[layer_idx].self_attn
    
    @staticmethod
    def get_mlp(model, layer_idx):
        return model.model.layers[layer_idx].mlp
    
    @staticmethod
    def N_LAYERS(model):
        return model.config.num_hidden_layers
    
    @staticmethod
    def N_QO_HEADS(model):
        return model.config.num_attention_heads
    
    @staticmethod
    def N_KV_HEADS(model):
        return model.config.num_key_value_heads