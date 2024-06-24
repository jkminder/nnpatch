import einops

def hidden_to_head(hidden, head_index):
    return einops.rearrange(hidden, "batch pos (head_index d_head) -> batch pos head_index d_head", head_index=head_index)

def head_to_hidden(head):
    return einops.rearrange(head, "batch pos head_index d_head -> batch pos (head_index d_head)")
