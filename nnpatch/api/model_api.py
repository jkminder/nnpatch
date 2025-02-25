from abc import ABC, abstractmethod


class ModelAPI(ABC):
    @staticmethod
    @abstractmethod
    def get_q(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_k(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_v(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_o(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_layer(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_attn(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def get_mlp(model, layer_idx):
        pass

    @staticmethod
    @abstractmethod
    def N_LAYERS(model):
        pass

    @staticmethod
    @abstractmethod
    def N_QO_HEADS(model):
        pass

    @staticmethod
    @abstractmethod
    def N_KV_HEADS(model):
        pass
