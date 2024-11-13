import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning import LightningModule, Trainer
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset
from functools import partial
from pyvene import TrainableIntervention, DistributedRepresentationIntervention
from nnsight import NNsight

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


class LowRankOrthogonalProjection(nn.Module):
    def __init__(self, embed_dim, rank=1, orthogonalize=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, rank), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

        self._P = None
        self._orthogonal_complement = None
        if orthogonalize:
            self = nn.utils.parametrizations.orthogonal(self)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

    def project(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

    def get_P(self):
        if self._P is None or self.training:
            self._P = torch.matmul(self.weight, self.weight.T)
        return self._P

    def get_orthogonal_complement(self, P=None):
        # recompute P
        P = self.get_P()
        if self._orthogonal_complement is None or self.training:
            I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
            self._orthogonal_complement = I - P
        return self._orthogonal_complement

    def forward(self, source, target):
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + P h_s
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(source.to(self.weight.dtype), P.T)

    def constant_forward(self, source_constant, target):
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + R s
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(source_constant.to(self.weight.dtype).unsqueeze(1), self.weight.T).squeeze(1)

    def __str__(self):
        return f"LowRankOrthogonalProjection()"

    @staticmethod
    def load_pretrained(path):
        state_dict = torch.load(path, weights_only=True)
        proj = LowRankOrthogonalProjection(state_dict["embed_dim"].item(), state_dict["rank"].item())
        state_dict.pop("embed_dim")
        state_dict.pop("rank")
        proj.load_state_dict(state_dict)
        return proj

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

class SubspaceIntervention(LightningModule):
    def __init__(self, model: nn.Module, projection: LowRankOrthogonalProjection, api: ModelAPI, layer: int, train_dataset, val_dataset, last_token_only=True, batch_size=32, num_workers=0, lr=1e-3, epochs=10):
        super().__init__()
        self.projection = projection
        self.model = NNsight(model)

        # disable model gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in projection.parameters():
            param.requires_grad = True

        self.train_dataset = train_dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.last_token_only = last_token_only
        self.layer = layer
        self.api = api
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
    
    def get_representation(self, value):
        if self.last_token_only:
            return value[:, -1, :]
        else:
            return value

    def forward(self, target_ids, target_mask, source_ids, source_mask, subspaces=None):
        # Collect Source and Target representations
        with self.model.trace(source_ids, attention_mask=source_mask) as tracer:
            source_representation = self.api.get_layer(self.model, self.layer).output[0].save()
            self.api.get_layer(self.model, self.layer).output.stop()
        source_representation = source_representation.value
        with self.model.trace(target_ids, attention_mask=target_mask) as tracer:
            target_representation = self.api.get_layer(self.model, self.layer).output[0]
            if self.last_token_only:
                self.api.get_layer(self.model, self.layer).output[0][:, -1, :] = self.projection(self.get_representation(source_representation), self.get_representation(target_representation))
            else:
                self.api.get_layer(self.model, self.layer).output[0] = self.projection(source_representation, target_representation)
            output = self.model.output.save()
        return output.value

    def __str__(self):
        return f"SubspaceIntervention(model={self.model})"

    def loss(self, output, labels):   
        last_token_logits = output[:, -1, :]
        labels = labels.to(last_token_logits.device)
        loss = F.cross_entropy(last_token_logits, labels)
        return loss

    def metrics(self, output, source_labels, target_labels):
        source_accuracy = (output[:, -1, :].argmax(dim=1) == source_labels).float().mean()
        target_accuracy = (output[:, -1, :].argmax(dim=1) == target_labels).float().mean()
        return {
            "source_accuracy": source_accuracy,
            "target_accuracy": target_accuracy,
        }

    def training_step(self, batch, batch_idx):
        target_ids, target_mask, source_ids, source_mask, source_labels, target_labels = batch
        output = self.forward(target_ids, target_mask, source_ids, source_mask).logits
        loss = self.loss(output, source_labels)
        metrics = self.metrics(output, source_labels, target_labels)
        metrics["loss"] = loss
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_ids, target_mask, source_ids, source_mask, source_labels, target_labels = batch
        output = self.forward(target_ids, target_mask, source_ids, source_mask).logits
        loss = self.loss(output, source_labels)
        metrics = self.metrics(output, source_labels, target_labels)
        self.log_dict(metrics, prog_bar=True)
        print(loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def configure_optimizers(self):
        params = [param for param in self.projection.parameters() if param.requires_grad]
        print(params)
        optimizer = torch.optim.AdamW(params=params, lr=self.lr)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0.1 * t_total, num_training_steps=t_total
        # )
        return optimizer

    # def on_before_backward(self, loss):
    #     print("on before backward")
    #     print("loss", loss)

    # def on_after_backward(self):
    #     print("on after backward")
    #     print("params grad", list(self.projection.parameters())[0][0].grad)
    #     print("grad", self.projection.weight)

    # def on_before_zero_grad(self, optimizer):
    #     print("238 after optimizer step", self.projection.weight)
    #     print("238 after optimizer step grad", self.projection.weight.grad)
    #     assert not torch.all(self.projection.weight == 0), "Zero weight ZG"

    # def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
    #     print("on train batch end", self.projection.weight)


    def backward(self, loss):
        loss.backward(retain_graph=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        print("optimizer step", self.projection.weight)
        print("grad", self.projection.weight.grad)
        print("")
        optimizer.step(closure=optimizer_closure)
        print("optimizer step done", self.projection.weight)
        assert not torch.all(self.projection.weight == 0), "Zero weight"
        
        
def create_tensor_dataset(source_tokens, target_tokens, source_label_index, target_label_index, source_attn_mask, target_attn_mask):
    return TensorDataset(target_tokens, target_attn_mask, source_tokens, source_attn_mask, source_label_index, target_label_index)

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

def train_projection_native(model: nn.Module, projection: LowRankOrthogonalProjection, api: ModelAPI, layer: int, train_dataset, val_dataset, batch_size=32, epochs=10, lr=1e-3, num_workers=0, **lightning_trainer_kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    intervention = SubspaceIntervention(model, projection, api, layer, train_dataset, val_dataset, batch_size=batch_size, epochs=epochs, lr=lr, num_workers=num_workers)
    intervention.train()
    model.train()
    # turn off model gradients
    trainer = Trainer(max_epochs=epochs, num_sanity_val_steps=0, *lightning_trainer_kwargs)
    print("training")
    trainer.fit(intervention)
    trainer.test(intervention)
    return

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
    from pyvene import IntervenableConfig, IntervenableModel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    intervention_type = "block_output"
    layer = 16

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