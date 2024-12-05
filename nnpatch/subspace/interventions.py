import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning import LightningModule, Trainer
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset
from functools import partial
from nnsight import NNsight
from huggingface_hub import PyTorchModelHubMixin
import warnings

from ..api.model_api import ModelAPI
torch.autograd.set_detect_anomaly(True)

def convert_statedict_from_pyvene(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    new_state_dict["rank"] = torch.tensor(1)
    for key, value in state_dict.items():
        if key.startswith("interchange_dim"):
            continue
        new_key = key.replace("", "")
        new_state_dict[new_key] = value
    return new_state_dict


class LowRankOrthogonalProjection(nn.Module, PyTorchModelHubMixin):
    """
    A projection that replaces the subspace (spanned by self.weight) value in the target activation with the one in the source activation.

    This is derived from the pyvene library and the implementation for https://arxiv.org/abs/2411.07404.
    """
    def __init__(self, embed_dim, rank=1, orthogonalize=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, rank), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

        self._P = None
        self._orthogonal_complement = None
        if orthogonalize:
            self = nn.utils.parametrizations.orthogonal(self)

    def project(self, x):
        """
        Computes the dot product of x and the subspace spanned by self.weight. This results in the value of x in the subspace spanned by self.weight.

        Args:
            x (torch.Tensor): The vector to project (batch_size, embed_dim)

        Returns:
            torch.Tensor: The value of x in the subspace spanned by self.weight (batch_size)
        """
        return torch.matmul(x.to(self.weight.dtype), self.weight)

    def get_P(self):
        """
        Returns the projection matrix P. This matrix will project any vector onto the subspace spanned by self.weight.
        """
        if self._P is None or self.training:
            self._P = torch.matmul(self.weight, self.weight.T)
        return self._P

    def get_orthogonal_complement(self, P=None):
        """
        Returns the projection matrix onto the orthogonal complement of the subspace spanned by self.weight. 
        """
        # recompute P
        P = self.get_P()
        if self._orthogonal_complement is None or self.training:
            I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
            self._orthogonal_complement = I - P
        return self._orthogonal_complement

    def forward(self, source, target):
        """
        Replace the subspace (spanned by self.weight) value in the target activation with the one in the source activation.

        Args:
            source (torch.Tensor): The source activation (batch_size, seq_len, embed_dim)
            target (torch.Tensor): The target activation (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: The modified target activation where the subspace value is replaced by the one from the source activation.
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + P h_s
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(source.to(self.weight.dtype), P.T)

    def constant_forward(self, source_constant, target):
        """
        Replace the subspace (spanned by self.weight) value in the target activation with a constant source value.
        If you are not training the subspace, make sure to call self.eval() before calling this function. 
        This will ensure that the projection matrix P is not updated during the forward pass.

        Args:
            source_constant (torch.Tensor): The constant source value or steering value - called c(w) in the paper (batch_size)
            target (torch.Tensor): The target activation (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: The modified target activation where the subspace value is replaced by the constant source value.
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + P u c(w) 
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(source_constant.to(self.weight.dtype).unsqueeze(1), self.weight.T).squeeze(1)

    def __str__(self):
        return f"LowRankOrthogonalProjection()"

    @staticmethod
    def load_pretrained(path):
        warnings.warn("Loading pretrained projection using load_pretrained() path is deprecated. Use the .from_pretrained() method instead.")
        state_dict = torch.load(path, weights_only=True)
        proj = LowRankOrthogonalProjection(state_dict["embed_dim"].item(), state_dict["rank"].item())
        state_dict.pop("embed_dim")
        state_dict.pop("rank")
        proj.load_state_dict(state_dict)
        return proj

    @staticmethod
    def from_subspace_basis(subspace_basis):
        """
        Create a projection from a subspace normal basis.

        Args:
            subspace_basis (torch.Tensor): The subspace basis (embed_dim, rank). Each column is a basis vector with unit norm.

        Returns:
            LowRankOrthogonalProjection: The projection that projects any vector onto the subspace spanned by subspace_basis.
        """
        assert len(subspace_basis.shape) == 2, "Subspace basis must be a 2D tensor of shape (embed_dim, rank)"
        rank = subspace_basis.shape[1]
        embed_dim = subspace_basis.shape[0]
        for basis in range(rank):
            assert subspace_basis[:, basis].norm() == 1, "Subspace basis must be normalized"
        proj = LowRankOrthogonalProjection(embed_dim, rank, orthogonalize=False)
        proj.weight = nn.Parameter(subspace_basis, requires_grad=True)
        proj = nn.utils.parametrizations.orthogonal(proj)
        return proj

def create_dataset(source_tokens, target_tokens, source_label_index, target_label_index, source_attn_mask, target_attn_mask, *args):
    # Create a dataset with the same structure as the original data
    dataset = Dataset.from_dict({
        'input_ids': target_tokens.tolist(),
        'attention_mask': target_attn_mask.tolist(),
        'source_input_ids': source_tokens.tolist(),
        'source_attention_mask': source_attn_mask.tolist(),
        'labels': torch.stack([source_label_index, target_label_index], dim=1).tolist(),
        'sources->base': [[target_tokens.shape[1] - 1]] * len(target_tokens),
        'subspaces': [0] * len(target_tokens),
    })
    return dataset


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = [b[k] for b in batch]
    return out


def compute_loss(logits, labels):
    last_token_logits = logits[:, -1, :]
    loss_fct = nn.CrossEntropyLoss()

    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(last_token_logits, labels[:, 0])
    
    return loss

def inputs_collator(inputs, device):
    for k, v in inputs.items():
        if "->" in k:
            inputs[k] = torch.tensor(v).to(device)
        elif "subspace" in k:
            inputs[k] = v
        elif v is not None:
            inputs[k] = torch.tensor(v).to(device)
    return inputs

def compute_metrics(eval_preds, eval_labels, tokenizer=None, return_target_accuracy=True, verbose=False,):
    total_count = 0
    correct_count = 0
    alter_correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        len_before = len(eval_label)
        eval_pred = eval_pred[eval_label[:,0] != eval_label[:,1]]
        eval_label = eval_label[eval_label[:,0] != eval_label[:,1]]
        len_after = len(eval_label)
        if len_before != len_after:
            print(f"Filtered {len_before - len_after} samples due to equal first token")
        pred_test_labels = torch.argmax(eval_pred[:, -1, :], dim=-1)
        correct_labels = eval_label[:, 0] == pred_test_labels
        alter_correct_count += (eval_label[:, 1] == pred_test_labels).sum().tolist()
        if verbose:
            for b_idx in range(len(eval_pred)):
                print("Pred:", tokenizer.decode(pred_test_labels[b_idx].item()), " Labels (source):", tokenizer.decode(eval_label[b_idx][0].item()), " Labels (target):", tokenizer.decode(eval_label[b_idx][1].item()))
                print("Correct:", correct_labels[b_idx].item())
                print("Alter Correct:", (eval_label[b_idx][1] == pred_test_labels[b_idx]).item())
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    alter_accuracy = round(alter_correct_count / total_count, 2)
    if return_target_accuracy:
        if alter_accuracy + accuracy > 1:
            print(alter_correct_count, correct_count, total_count, accuracy, alter_accuracy)
            raise ValueError("Accuracy is greater than 1")
        return {"accuracy": accuracy, "target_accuracy": alter_accuracy}
    return {"accuracy": accuracy}


def train_projection(model: nn.Module, projection: LowRankOrthogonalProjection, layer, train_dataset, val_dataset, batch_size=32, epochs=10, lr=1e-3, num_workers=4, device="cuda"):
    try:
        from pyvene import TrainableIntervention, DistributedRepresentationIntervention
        from pyvene import IntervenableConfig, IntervenableModel
    except ImportError:
        raise ImportError("Training a projection is currently still based on pyvene. Please install pyvene with `pip install pyvene` to use this feature.")

    
    class InterchangeIntervention(TrainableIntervention, DistributedRepresentationIntervention):
        def __init__(self, projection: LowRankOrthogonalProjection, layer, device=None, last_token_only=True, **kwargs):
            super().__init__(**kwargs)
            self.proj = projection
            self.layer = layer
            self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.last_token_only = last_token_only
            self.proj.to(self.device)
            self._activated = False
            self._source=False

        def forward(self, target, source, subspaces=None):
            # h_t = (I-P) h_s + P h_t
            intervened_target = self.proj(source, target)
            return intervened_target.to(target.dtype)


    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    intervention_type = "block_output"

    config = IntervenableConfig([
        {
            "layer": layer,
            "component": intervention_type,
            "intervention_type": lambda **kwargs: InterchangeIntervention(projection, **kwargs),
        },
    ])

    intervenable = IntervenableModel(config, model)
    intervenable.disable_model_gradients()
    key = list(intervenable.interventions.keys())[0]
    try:
        intervenable.set_device(device)
    except Exception as e:
        # TO DEVICE for 4bit
        intervenable.interventions[key][0].to(device)
        intervenable.interventions[key][0].rotate_layer.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    print("Training")
    print("Train dataset size", len(train_dataloader))
    intervenable.train_alignment(
        train_dataloader=train_dataloader,
        compute_loss=compute_loss,
        compute_metrics=partial(compute_metrics, verbose=False),
        inputs_collator=partial(inputs_collator, device=device),
        epochs=epochs,
        lr=lr
    )

    print("Validation")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    print("Validation dataset size", len(val_dataloader))
    metrics = intervenable.eval_alignment(
        eval_dataloader=val_dataloader,
        compute_metrics=partial(compute_metrics, verbose=False),
        inputs_collator=partial(inputs_collator, device=device),
    )
    print("Validation metrics", metrics)

    proj = intervenable.interventions[key][0]
    return proj
