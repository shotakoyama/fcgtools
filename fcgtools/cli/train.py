from argparse import ArgumentParser
from fcgtools.train.opter import Opter
from fcgtools.train.trainer import Trainer
from fcgtools.train.loader import (
        load_bart_loaders,
        load_gpt2_loaders)
from fcgtools.train.model import (
        load_bart_model,
        load_gpt2_model)
from fcgtools.train.losscalc import (
        BARTLossCalc,
        GPT2LossCalc)


def train_main(args, model, train_loader, valid_loader, losscalc_class):
    model.cuda()

    opter = Opter(
            model,
            args.lr,
            max_grad_norm = args.max_grad_norm,
            scheduler = args.scheduler,
            warmup_steps = args.warmup_steps,
            start_factor = args.start_factor,
            weight_decay = args.weight_decay)

    trainer = Trainer(
            train_loader,
            valid_loader,
            model,
            opter,
            losscalc_class,
            args.epochs,
            args.step_interval,
            args.save_interval)

    trainer.run()


def bart_main(args):
    train_loader, valid_loader = load_bart_loaders(args)
    model = load_bart_model(args.arch)
    train_main(args, model, train_loader, valid_loader, BARTLossCalc)


def gpt2_main(args):
    train_loader, valid_loader = load_gpt2_loaders(args)
    model = load_gpt2_model(args.arch)
    train_main(args, model, train_loader, valid_loader, GPT2LossCalc)


def parse_bart_args(sub_parsers):
    parser = sub_parsers.add_parser('bart')
    parser.add_argument('data')
    parser.add_argument('--arch', default = 'base')
    parser.add_argument('--max-tokens', type = int, default = 4000)
    parser.add_argument('--lr', type = float, default = 0.00005)
    parser.add_argument('--scheduler', default = 'constant')
    parser.add_argument('--warmup-steps', type = int, default = 1)
    parser.add_argument('--start-factor', type = float, default = 1.0)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--weight-decay', type = float, default = 0.0)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--step-interval', type = int, default = 1)
    parser.add_argument('--save-interval', type = int, default = 1)
    parser.set_defaults(handler = bart_main)


def parse_gpt2_args(sub_parsers):
    parser = sub_parsers.add_parser('gpt2')
    parser.add_argument('data')
    parser.add_argument('--arch', default = 'small')
    parser.add_argument('--max-tokens', type = int, default = 4000)
    parser.add_argument('--lr', type = float, default = 0.00005)
    parser.add_argument('--scheduler', default = 'constant')
    parser.add_argument('--warmup-steps', type = int, default = 1)
    parser.add_argument('--start-factor', type = float, default = 1.0)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--weight-decay', type = float, default = 0.0)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--step-interval', type = int, default = 1)
    parser.add_argument('--save-interval', type = int, default = 1)
    parser.add_argument('--sep', default = ' #')
    parser.add_argument('--backward-only-target', action = 'store_true')
    parser.set_defaults(handler = gpt2_main)


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    parse_bart_args(sub_parsers)
    parse_gpt2_args(sub_parsers)
    args = parser.parse_args()
    print(args)
    args.handler(args)

