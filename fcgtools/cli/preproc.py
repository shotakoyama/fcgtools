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
            args.add_gector_tag,
            args.gector_tag_sep,
            args.source_detokenize,
            args.target_with_initial_space)

    if args.train is not None:
        preproc(
            args.train,
            'train',
            size = args.size,
            raw = args.raw,
            tag_path = args.train_tag)

    if args.valid is not None:
        preproc(
            args.valid,
            'valid',
            raw = args.raw,
            tag_path = args.valid_tag)

    if args.test is not None:
        preproc(
            args.test,
            'test',
            raw = args.raw,
            only_source = True,
            tag_path = args.test_tag)


def bart_main(args):
    tokenizer = BARTWrapTokenizer(args.arch)
    preproc_main(args, tokenizer)


def gpt2_main(args):
    tokenizer = GPT2WrapTokenizer(args.arch)
    preproc_main(args, tokenizer)


def parse_common_args(parser):
    parser.add_argument('--train', default = None)
    parser.add_argument('--valid', default = None)
    parser.add_argument('--test', default = None)
    parser.add_argument('--train-tag', default = None)
    parser.add_argument('--valid-tag', default = None)
    parser.add_argument('--test-tag', default = None)
    parser.add_argument('--left', default = '<<')
    parser.add_argument('--right', default = '>>')
    parser.add_argument('--size', type = int, default = None)
    parser.add_argument('--add-gector-tag', action = 'store_true')
    parser.add_argument('--gector-tag-sep', default = ' // ')
    parser.add_argument('--source-detokenize', action = 'store_true')
    parser.add_argument('--target-with-initial-space', action = 'store_true')
    parser.add_argument('--raw', action = 'store_true')


def parse_bart_args(sub_parsers):
    parser = sub_parsers.add_parser('bart')
    parse_common_args(parser)
    parser.add_argument('--arch', default = 'base')
    parser.set_defaults(handler = bart_main)


def parse_gpt2_args(sub_parsers):
    parser = sub_parsers.add_parser('gpt2')
    parse_common_args(parser)
    parser.add_argument('--arch', default = 'small')
    parser.set_defaults(handler = gpt2_main)


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    parse_bart_args(sub_parsers)
    parse_gpt2_args(sub_parsers)
    args = parser.parse_args()
    args.handler(args)

