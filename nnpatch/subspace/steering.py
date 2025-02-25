import torch


class SteerHook:
    def __init__(self, proj, layer, value=None, device=None, last_token_only=True):
        self.proj = proj
        self.value = value
        self.layer = layer
        self.hook = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_token_only = last_token_only
        self.proj.to(self.device)
        self._activated = True

    def set_value(self, value):
        print("setting value", value)
        self.value = value
        self.activate()

    def get_value(self, output):
        return self.value.to(self.device)

    def __call__(self, module, input, output):
        if not self._activated:
            return output
        value = self.get_value(output)
        assert self.hook is not None, "Hook is not set"
        assert value is not None, "Value is not set"
        assert (
            value.shape[0] == output[0].shape[0]
        ), f"Value shape {value.shape} does not match input shape {input.shape}"
        if self.last_token_only:
            target = output[0][:, -1, :]
        else:
            target = output[0]

        source_value = self.get_value(input)
        # h_t = (I-P) h_t + P h_s
        intervened_target = self.proj.constant_forward(source_value, target)

        if self.last_token_only:
            output[0][:, -1, :] = intervened_target
        else:
            output[0][:, :, :] = intervened_target
        return output

    def remove(self):
        assert self.hook is not None, "Hook is not set"
        self.hook.remove()
        self.hook = None

    def attach(self, model):
        assert self.hook is None, "Hook is already set"
        base_model = model.model
        if not hasattr(base_model, "layers"):
            base_model = base_model.model
        self.hook = base_model.layers[self.layer].register_forward_hook(self)
        return self.hook

    def deactivate(self):
        self._activated = False

    def activate(self):
        self._activated = True


class BinaryHook(SteerHook):
    def __init__(self, proj, layer, value_a=6.0, value_b=-6.0, last_token_only=True):
        super().__init__(proj, layer, last_token_only=last_token_only)
        self.value_a = value_a
        self.value_b = value_b
        self.value = self.value_a

    def get_value(self, output):
        if isinstance(self.value, (int, float)):
            batch_size = output[0].shape[0]
            return torch.tensor(self.value, dtype=torch.float32).repeat(batch_size).to(self.device)
        else:
            return self.value.to(self.device)

    def set_constant_a(self):
        self.value = self.value_a
        self.activate()

    def set_constant_b(self):
        self.value = self.value_b
        self.activate()

    def set_binary(self, value):
        """
        Value is a boolean tensor of shape (batch_size). True means value_b, False means value_a.
        """
        value = value.to(torch.float32)
        self.value = self.value_b * value + self.value_a * (1 - value)
        self.activate()


class FeatureCollectionHook:
    def __init__(self, proj, layer, value=None, device=None, last_token_only=True, only_one_forward=True):
        self.proj = proj
        self.layer = layer
        self.hook = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_token_only = last_token_only
        self.proj.to(self.device)
        self.only_one_forward = only_one_forward
        self._activated = True
        self._features = []

    def __call__(self, module, input, output):
        if not self._activated:
            return output
        assert self.hook is not None, "Hook is not set"

        if self.last_token_only:
            target = output[0][:, -1, :]
        else:
            target = output[0]

        self._features.append(self.proj.project(target))
        if self.only_one_forward:
            self.remove()
        return output

    def remove(self):
        assert self.hook is not None, "Hook is not set"
        self.hook.remove()
        self.hook = None

    def attach(self, model):
        assert self.hook is None, "Hook is already set"
        base_model = model.model
        if not hasattr(base_model, "layers"):
            base_model = base_model.model
        self.hook = base_model.layers[self.layer].register_forward_hook(self)
        return self.hook

    @property
    def features(self):
        return torch.cat(self._features, dim=0)
