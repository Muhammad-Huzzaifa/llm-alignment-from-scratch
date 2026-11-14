import torch


def token_logprobs(model, input_ids, attention_mask, prompt_lens):
    """compute the logprobs of each token,
    generate summary mask which is equal to 1 only where there are summary tokens
    
    Args:
        model (AutoModelForCausalLM): the model to compute logprobs with
        input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids
        attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens and 0s for padding
        prompt_lens (torch.Tensor): tensor of shape (batch,) with the lengths of the prompts

    Returns:
        tuple: (tok_logprobs, summary_mask) 
            where tok_logprobs is a tensor of shape (batch, seq_len-1) with the logprobs of each token, 
            and summary_mask is a tensor of shape (batch, seq_len-1) with 1s for tokens in the summary and 0s for tokens in the prompt or padding
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (batch, seq_len, d_model)
    shifted_logits = logits[:, :-1, :]  # (batch, seq_len-1, d_model)
    shifted_labels = input_ids[:, 1:]  # (batch, seq_len-1)
    shifted_attn = attention_mask[:, 1:]  # (batch, seq_len-1)

    b, s, v = shifted_logits.shape
    tok_logprobs = -torch.nn.functional.cross_entropy(
        shifted_logits.float().reshape(-1, v), shifted_labels.reshape(-1), reduction="none"
    ).view(b, s)  # (batch, seq_len-1)

    summary_mask = torch.zeros_like(shifted_labels, dtype=torch.bool)  # (batch, seq_len-1)
    for i, pl in enumerate(prompt_lens.tolist()):
        start = pl - 1  # due to shifted labels
        summary_mask[i, start:] = shifted_attn[i, start:]

    return tok_logprobs, summary_mask


def sequence_logprobs(model, input_ids, attention_mask, prompt_lens):
    """summition of the logprobs of the summary tokens only, 
    for each sequence in the batch

    Args:
        model (AutoModelForCausalLM): the model to compute logprobs with
        input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids
        attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens and 0s for padding
        prompt_lens (torch.Tensor): tensor of shape (batch,) with the lengths of the prompts
    
    Returns:
        torch.Tensor: tensor of shape (batch,) with the sum of logprobs of the summary tokens for each sequence in the batch
    """
    lp, mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)
    return (lp * mask).sum(dim=1) # (batch,)


def last_real_index(attention_mask):
    """get the last token's index which is not pad, handling the left or right pad

    Args:
        attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens and 0s for padding
    
    Returns:
        torch.Tensor: tensor of shape (batch,) with the index of the last real token for each sequence in the batch   
    """
    L = attention_mask.size(1)
    positions = torch.arange(L, device=attention_mask.device)
    return (positions * attention_mask.long()).max(dim=-1).values


def generate_summaries(model, tok, input_ids, attention_mask, max_new_tokens, do_sample=False, temperature=1.0, top_k=0, top_p=1.0):
    """generate summaries given the prompt tokens, 
    using the model's generate method with the specified parameters

    Args:
        model (AutoModelForCausalLM): the model to generate summaries with
        tok (AutoTokenizer): the tokenizer used for encoding and decoding
        input_ids (torch.Tensor): tensor of shape (batch, seq_len) with token ids of the prompts
        attention_mask (torch.Tensor): tensor of shape (batch, seq_len) with 1s for real tokens and 0s for padding
        max_new_tokens (int): maximum number of new tokens to generate
        do_sample (bool): whether to use sampling or greedy decoding
        temperature (float): temperature for sampling
        top_k (int): top-k sampling parameter
        top_p (float): top-p sampling parameter

    Returns:
        torch.Tensor: tensor of shape (batch, gen_len) with the generated token ids for each sequence in the batch
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:  # sampling-only flags are invalid during greedy decoding
        gen_kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    ) # (batch, seq_len+gen_len)
    return out[:, input_ids.size(1):]  # (batch, gen_len)