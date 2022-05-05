from argparse import ArgumentParser
from fcgtools.preproc.tokenizer import (
        BARTWrapTokenizer,
        GPT2WrapTokenizer)
from fcgtools.preproc.preprocessor import Preprocessor

def preproc_main(args, tokenizer):
    preproc = Preprocessor(
            args.left,
            args.right,
            tokenizer,
            args.source_detokenize,
            args.target_with_initial_space)

    if args.train is not None:
        preproc(args.train, 'train', size = args.size, raw = args.raw)

    if args.valid is not None:
        preproc(args.valid, 'valid', raw = args.raw)

    if args.test is not None:
        preproc(args.test, 'test', raw = args.raw, only_source = True)


def bart_main(args):
    tokenizer = BARTWrapTokenizer(args.arch)
    preproc_main(args, tokenizer)


def gpt2_main(args):
    tokenizer = GPT2WrapTokenizer(args.arch)
    preproc_main(args, tokenizer)


def parse_bart_args(sub_parsers):
    parser = sub_parsers.add_parser('bart')
    parser.add_argument('--train', default = None)
    parser.add_argument('--valid', default = None)
    parser.add_argument('--test', default = None)
    parser.add_argument('--arch', default = 'base')
    parser.add_argument('--left', default = '<<')
    parser.add_argument('--right', default = '>>')
    parser.add_argument('--size', type = int, default = None)
    parser.add_argument('--source-detokenize', action = 'store_true')
    parser.add_argument('--target-with-initial-space', action = 'store_true')
    parser.add_argument('--raw', action = 'store_true')
    parser.set_defaults(handler = bart_main)


def parse_gpt2_args(sub_parsers):
    parser = sub_parsers.add_parser('gpt2')
    parser.add_argument('--train', default = None)
    parser.add_argument('--valid', default = None)
    parser.add_argument('--test', default = None)
    parser.add_argument('--arch', default = 'small')
    parser.add_argument('--left', default = '<<')
    parser.add_argument('--right', default = '>>')
    parser.add_argument('--size', type = int, default = None)
    parser.add_argument('--source-detokenize', action = 'store_true')
    parser.add_argument('--target-with-initial-space', action = 'store_true')
    parser.add_argument('--raw', action = 'store_true')
    parser.set_defaults(handler = gpt2_main)


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    parse_bart_args(sub_parsers)
    parse_gpt2_args(sub_parsers)
    args = parser.parse_args()
    args.handler(args)

