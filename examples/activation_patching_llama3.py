import sys
sys.path.append("..")

import torch
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight.model import NNsight

from nnpatch import Sites, act_patch
from nnpatch.api.llama import Llama3

device = "cuda:0"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


TRAIN_DATA = ...
VAL_DATA = ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", default=0, type=int)
    parser.add_argument("--query-range", default=8, type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compute-heads", action="store_true")
    parser.add_argument("--layers", default=None, type=int, nargs="+")
    parser.add_argument("--seq-pos-type", default="lastk")
    parser.add_argument("--seq-pos", default=25, nargs="+", type=int)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    print(args)
    assert not (args.context_info_flow and (args.context_to_prior or args.prior_to_context)), "Cannot have both context info flow and context to prior"
    assert not (args.context_to_prior and args.prior_to_context), "Cannot have both context to prior and prior to context"

    val_data = pd.read_csv(VAL_DATA)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    nnmodel = NNsight(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    
    val_slice = val_data.iloc[args.dataset_index: args.dataset_index+args.query_range]
    
    # Since the last token in the prompt is \n, we make sure to get the tokenized version of '\n answer', which can be different from just the answer tokenized.
    source_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in val_slice.answer]).to(device)
    target_answer_index = torch.tensor([tokenizer.encode("\n" + a)[1] for a in val_slice.answer]).to(device)

    source_prompts = val_slice.source.tolist()
    target_prompts = val_slice.target.tolist()
    

    # We further assume that both source and target prompts are the same length.
    source_prompts = tokenizer(source_prompts, return_tensors="pt", padding=True)
    attention_mask = source_prompts["attention_mask"].to(device)
    source_prompts = source_prompts["input_ids"].to(device)
    target_prompts = tokenizer(target_prompts, return_tensors="pt", padding=True)["input_ids"].to(device)
    
    if args.compute_heads:
        site_names = ["o", "k", "q", "v"]
    elif args.all:
        site_names = ["resid", "attn", "mlp", "o", "k", "q", "v"]
    else:
        site_names = ["resid", "attn", "mlp"]
    
    seq_pos = args.seq_pos
        
    sites = Sites(site_names=site_names, layers=args.layers, seq_pos=seq_pos, seq_pos_type=args.seq_pos_type)
    
    out = act_patch(nnmodel, Llama3, sites, source_prompts, target_prompts, source_answer_index, target_answer_index, attention_mask=attention_mask)
    

    torch.save({
        'patching_results': out,
        'source_prompt': source_prompts,
        'target_prompt': target_prompts,
        }, args.name + ".pt"
    )    