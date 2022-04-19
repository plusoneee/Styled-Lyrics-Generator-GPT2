import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_trained_model(model, tokenizer, checkpoint_dir='path/to/tunemodels/dir'):
    checkpoint_filepath = checkpoint_dir+os.sep+'checkpoint.bin'
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint)
    tokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


if __name__ == '__main__':
    special_tokens_dict = {
        'additional_special_tokens':['[s:emo]', '[e:emo]', '[s:lyrics]', '[e:lyrics]']
    }

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')
    tokenizer.add_special_tokens(special_tokens_dict)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model, tokenizer = load_trained_model(model, tokenizer)

    print(model)