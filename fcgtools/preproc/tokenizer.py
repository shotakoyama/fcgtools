from transformers import (
        BartTokenizer,
        GPT2Tokenizer)

class WrapTokenizer:

    def tokenize(self, text):
        text = self.tokenizer.encode(text, add_special_tokens = False)
        return text


def load_bart_tokenizer(arch):
    arch_name = 'facebook/bart-{}'.format(arch)
    tokenizer = BartTokenizer.from_pretrained(arch_name)
    return tokenizer


def load_gpt2_tokenizer(arch):
    if arch == 'small':
        arch_name = 'gpt2'
    else:
        arch_name = 'gpt2-{}'.format(arch)
    tokenizer = GPT2Tokenizer.from_pretrained(arch_name)
    return tokenizer


class BARTWrapTokenizer(WrapTokenizer):

    def __init__(self, arch):
        self.tokenizer = load_bart_tokenizer(arch)


class GPT2WrapTokenizer(WrapTokenizer):

    def __init__(self, arch):
        self.tokenizer = load_gpt2_tokenizer(arch)

