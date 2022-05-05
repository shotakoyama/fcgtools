from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fcgtools.preproc.data import MemmapData
from fcgtools.preproc.tokenizer import (
        load_bart_tokenizer,
        load_gpt2_tokenizer)
from fcgtools.util.dataset import Dataset
from fcgtools.util.collator import (
        BARTCollator,
        GPT2Collator)
from fcgtools.util.sampler import (
        FixedSampler,
        RandomSampler)

from transformers import BartTokenizer

def load_dataset(data, phase):
    src_data = MemmapData(Path(data) / phase / 'src')
    trg_data = MemmapData(Path(data) / phase / 'trg')
    dataset = Dataset(src_data, trg_data)
    return dataset


def load_source_dataset(data, phase):
    src_data = MemmapData(Path(data) / phase / 'src')
    dataset = Dataset(src_data)
    return dataset


def load_tokenizer(arch):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-{}'.format(arch))
    return tokenizer


def load_loaders(args, collator):
    train_dataset = load_dataset(args.data, 'train')
    valid_dataset = load_dataset(args.data, 'valid')

    train_sampler = RandomSampler(
            train_dataset,
            args.max_tokens)
    valid_sampler = FixedSampler(
            valid_dataset,
            args.max_tokens)

    train_loader = DataLoader(
            train_dataset,
            batch_sampler = train_sampler,
            collate_fn = collator)

    valid_loader = DataLoader(
            valid_dataset,
            batch_sampler = valid_sampler,
            collate_fn = collator)

    return train_loader, valid_loader


def load_bart_loaders(args):
    tokenizer = load_bart_tokenizer(args.arch)
    collator = BARTCollator(tokenizer)
    return load_loaders(args, collator)


def load_gpt2_loaders(args):
    tokenizer = load_gpt2_tokenizer(args.arch)
    collator = GPT2Collator(
            tokenizer,
            sep = args.sep,
            backward_only_target = args.backward_only_target)
    return load_loaders(args, collator)

