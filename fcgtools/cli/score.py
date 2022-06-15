from argparse import ArgumentParser
from fcgtools.score import (
        fcg_sent_bleu,
        fcg_corpus_score)

def parser_args():
    parser = ArgumentParser()
    parser.add_argument(
            '-y',
            '--hyp',
            '--hypothesis',
            dest = 'hyp',
            required = True)
    parser.add_argument(
            '-r',
            '--ref',
            '--reference',
            dest = 'ref',
            required = True)
    parser.add_argument(
            '-t',
            '--text-ref',
            '--text-reference',
            dest = 'textref',
            action = 'store_true')
    parser.add_argument(
            '-s',
            '--sent-level',
            '--sentence-level',
            dest = 'sentlevel',
            action = 'store_true')
    parser.add_argument(
            '-o',
            '--order-by-bleu',
            dest = 'order',
            action = 'store_true')
    parser.add_argument(
            '-v',
            '--verbose',
            dest = 'verbose',
            action = 'store_true')
    return parser.parse_args()


def load_inputs(hyp_path, ref_path, textref):

    with open(hyp_path) as hf:
        hyp_list = [hyp.strip() for hyp in hf]

    with open(ref_path) as rf:
        if textref:
            ref_list = [ref.strip() for ref in rf]
        else:
            ref_list = [ref.strip().split('\t')[-1] for ref in rf]

    return hyp_list, ref_list


def corpus_level(hyp_list, ref_list):
    f = fcg_corpus_score(hyp_list, ref_list)
    print('{:.2f}'.format(f))


def sent_level(hyp_list, ref_list, order_by_bleu, verbose):

    bleu_list = [
        (i, fcg_sent_bleu(hyp, ref))
        for i, (hyp, ref)
        in enumerate(zip(hyp_list, ref_list))]

    if order_by_bleu:
        bleu_list.sort(key = lambda x: x[1])

    for i, bleu in bleu_list:
        if verbose:
            print('{}\t{:.2f}'.format(i + 1, bleu))
            print('hyp: {}'.format(hyp_list[i]))
            print('ref: {}'.format(ref_list[i]))
        else:
            print('{:.2f}'.format(bleu))


def main():
    args = parser_args()

    hyp_list, ref_list = load_inputs(args.hyp, args.ref, args.textref)

    if args.sentlevel:
        sent_level(hyp_list, ref_list, args.order, args.verbose)
    else:
        corpus_level(hyp_list, ref_list)

