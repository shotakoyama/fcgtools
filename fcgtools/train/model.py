from transformers import (
        BartForConditionalGeneration,
        GPT2LMHeadModel)

def load_bart_model(arch):
    arch = 'facebook/bart-{}'.format(arch)
    model = BartForConditionalGeneration.from_pretrained(arch)
    return model


def load_gpt2_model(arch, pad_token_id = None):
    if arch == 'small':
        arch_name = 'gpt2'
    else:
        arch_name = 'gpt2-{}'.format(arch)

    model = GPT2LMHeadModel.from_pretrained(arch_name, pad_token_id = pad_token_id)
    return model

