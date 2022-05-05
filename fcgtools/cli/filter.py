from argparse import ArgumentParser
import re
from lemminflect import getAllInflections

def parse_args():
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
    return parser.parse_args()


def word_quote_check(source, query):
    candidates = [query] + [token for lst in getAllInflections(query).values() for token in lst]
    return any(token in source for token in candidates)


def sent_quote_check(hyp, ref):

    quoted_tokens = re.findall(r'(?<=<<)[^ <>]+(?=>>)', hyp)
    quoted_tokens = [token.lower() for token in quoted_tokens]
    quoted_tokens = list(set(quoted_tokens))

    source_tokens = set([token.lower() for token in ref.split()])

    return all(word_quote_check(source_tokens, token) for token in quoted_tokens)


def main():
    args = parse_args()

    with open(args.hyp) as f:
        hyps = [x.strip() for x in f]

    with open(args.ref) as f:
        refs = [x.strip().split('\t')[0] for x in f]

    for hyp, ref in zip(hyps, refs):

        if not sent_quote_check(hyp, ref):
            print('<NO_COMMENT>')
        else:
            print(hyp)

