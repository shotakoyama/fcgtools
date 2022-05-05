from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from fcgtools.preproc.tokenizer import (
        load_bart_tokenizer,
        load_gpt2_tokenizer)
from fcgtools.train.loader import load_source_dataset
from fcgtools.train.model import (
        load_bart_model,
        load_gpt2_model)

def make_bart_input(dct, bos_id, eos_id):
    sent = [bos_id] + dct['source'] + [eos_id]
    return sent


def make_gpt2_input(dct, eos_id, sep_ids):
    sent = [eos_id] + dct['source'] + sep_ids
    return sent


def bart_postproc(sent, bos_id, eos_id):
    sent = sent.tolist()
    assert sent[0] == eos_id
    assert sent[1] == bos_id
    assert sent[-1] == eos_id

    # Sometimes 2 or more bos may be at the beginning of the sentence.
    # For example, </s> <s> <s> This is a feedback comment.
    # This strange thing can happen when using BART-large, not with BART-base.
    # So this script removes all <s> and </s> in input sentence.
    sent = [x for x in sent if x not in {bos_id, eos_id}]
    return sent


def bart_main(args):
    tokenizer = load_bart_tokenizer(args.arch)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    model = load_bart_model(args.arch)
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    model.cuda()

    dataset = load_source_dataset(args.srcdata, args.phase)
    for dct in dataset:
        sent = make_bart_input(dct, bos_id, eos_id)
        pred = model.generate(
            torch.tensor([sent]).cuda(),
            max_length = 512,
            num_beams = args.beam)
        sent = bart_postproc(pred[0], bos_id, eos_id)
        sent = tokenizer.decode(sent)
        print(sent)


def gpt2_main(args):
    tokenizer = load_gpt2_tokenizer(args.arch)
    eos_id = tokenizer.eos_token_id
    sep_ids = tokenizer.encode(args.sep)
    model = load_gpt2_model(args.arch, pad_token_id = eos_id)
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    model.cuda()

    dataset = load_source_dataset(args.srcdata, args.phase)
    for dct in dataset:
        sent = make_gpt2_input(dct, eos_id, sep_ids)
        pred = model.generate(
                torch.tensor([sent]).cuda(),
                max_length = 512,
                num_beams = args.beam)
        sent = tokenizer.decode(pred[0])
        sent = sent.split(args.sep.strip())[-1].rstrip('<|endoftext|>').strip()
        print(sent)

def parse_bart_args(sub_parsers):
    parser = sub_parsers.add_parser('bart')
    parser.add_argument('srcdata')
    parser.add_argument('--phase', default = 'valid')
    parser.add_argument('--arch', default = 'base')
    parser.add_argument('--beam', type = int, default = 5)
    parser.add_argument('--checkpoint')
    parser.set_defaults(handler = bart_main)


def parse_gpt2_args(sub_parsers):
    parser = sub_parsers.add_parser('gpt2')
    parser.add_argument('srcdata')
    parser.add_argument('--phase', default = 'valid')
    parser.add_argument('--arch', default = 'small')
    parser.add_argument('--beam', type = int, default = 5)
    parser.add_argument('--checkpoint')
    parser.add_argument('--sep', default = ' #')
    parser.set_defaults(handler = gpt2_main)


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    parse_bart_args(sub_parsers)
    parse_gpt2_args(sub_parsers)
    args = parser.parse_args()
    args.handler(args)

