# nnpatch
nnpatch is a Python library designed for neural network model interpretability through activation patching. It is based on [nnsight](https://nnsight.net) and inspired by https://github.com/callummcdougall/path_patching.

The library builds on the concept of a Site. A Site is any position in a model where we can patch. This can be a specific token position, a specific head, a specific layer, multiple positions at once, a specific block (mlp, attn, resid) or any combination of those.

Currently `mlp`, `attn` and `resid` refer to the post block activations.

# Installation
```
pip install git+https://github.com/jkminder/nnpatch
```

# Quick Start

## Automated Activation Patching
```python
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnpatch import Sites, act_patch, Site
from nnpatch.api.llama import Llama3

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-id")
tokenizer = AutoTokenizer.from_pretrained("your-model-id")
tokenizer.padding_side = "left"

# Prepare your data
data = pd.read_csv("path/to/your/validation.csv")



# Since the last token in the prompt is \n, we make sure to get the tokenized version of '\n answer', which can be different from just the answer tokenized.
source_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in data.source_answers]).to(device)
target_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in data.target_answers]).to(device)

source_prompts = data.source_prompts.tolist()
target_prompts = data.target_prompts.tolist()

# We assume that source and target are the same length (otherwise make sure the two are padded to the same length)
source_prompts = tokenizer(source_prompts, return_tensors="pt", padding=True)
attention_mask = source_prompts["attention_mask"].to(device)
source_prompts = source_prompts["input_ids"].to(device)
target_prompts = tokenizer(target_prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

site_names = ["resid", "attn", "mlp", "o", "k", "q", "v"]

# Define patch sites and layers
# We patch on layers 1 to 4, set to None to patch on all layers
# Patch on each position
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="each")
# Patch on all positions at once
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type=None)
# Patch on the last token
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="last")
# Patch on the last k=10 tokens
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="lastk", seq_pos=10)
# Patch on a list of specific positions at once (replace all these positions in one forward pass)
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="custom_constant", seq_pos=[-3,-102])
# Patch on a list of specific positions each 
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="custom", seq_pos=[-3,-102])

# Apply activation patching
out = act_patch(nnmodel, Llama3, sites, source_prompts, target_prompts, source_answer_index, target_answer_index, attention_mask=attention_mask)
# out: Dict of site_name: tensor
```

## Do custom things


```python
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnpatch import Sites, act_patch
from nnpatch.api.llama import Llama3

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-id")
tokenizer = AutoTokenizer.from_pretrained("your-model-id")
tokenizer.padding_side = "left"

# Prepare your data
data = pd.read_csv("path/to/your/validation.csv")



# Since the last token in the prompt is \n, we make sure to get the tokenized version of '\n answer', which can be different from just the answer tokenized.
source_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in data.source_answers]).to(device)
target_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in data.target_answers]).to(device)

source_prompts = data.source_prompts.tolist()
target_prompts = data.target_prompts.tolist()

# We assume that source and target are the same length (otherwise make sure the two are padded to the same length)
source_prompts = tokenizer(source_prompts, return_tensors="pt", padding=True)
attention_mask = source_prompts["attention_mask"].to(device)
source_prompts = source_prompts["input_ids"].to(device)
target_prompts = tokenizer(target_prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

site_names = ["resid", "attn", "mlp", "o", "k", "q", "v"]

# Define patch sites and layers
# We patch on layers 1 to 4, set to None to patch on all layers
# Patch on each position
sites = Sites(site_names=site_names, layers=[1,2,3,4], seq_pos_type="each")

# Dict of site_name: List of Sites
sites = sites.get_sites_dict(nnmodel, Llama3, source_tokens)
# or just create the sites yourself.
sites = [Site.get_site(Llama3, site_name="q", layer=12, head=23, seq_pos=[-1])]

# Cache Run
with nnmodel.trace(source_prompts, attention_mask=attention_mask) as invoker:
    for site in sites:
        site.cache(nnmodel)

# Patch run
with nnmodel.trace(target_prompts, attention_mask=attention_mask) as invoker:
    for site in sites:
        site.patch(nnmodel)


```
