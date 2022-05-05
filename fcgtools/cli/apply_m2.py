from argparse import ArgumentParser
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
            '-t',
            '--tsv',
            dest = 'tsv',
            required = True)
    parser.add_argument(
            '-m',
            '--m2',
            dest = 'm2',
            required = True)
    return parser.parse_args()


class Edit:

    def __init__(self, line):
        line = line.split('|||')
        self.error_type = line[1]
        span = line[0].split()[1:]
        self.start = int(span[0])
        self.end = int(span[1])
        self.cor = line[2].split()


class SentencePair:

    def __init__(self, sent):
        sent = sent.split('\n')
        self.src = sent[0].split()[1:]

        edits = [Edit(edit) for edit in sent[1:]]
        edits = [edit for edit in edits if edit.error_type not in {'noop', 'UNK', 'Um'}]
        self.edits = edits


def load_m2(m2_path):
    with open(m2_path) as f:
        m2 = f.read().strip().split('\n\n')
    m2 = [SentencePair(sent) for sent in m2]
    return m2


def load_tsv(ref_path):
    data = []
    with open(ref_path) as f:

        for line in f:
            tup = line.strip().split('\t')

            if len(tup) == 3: # for test
                source, start, end = tup
                feedback = None
            elif len(tup) == 4: # for train/valid
                source, start, end, feedback = tup
            else:
                assert False

            source = source.split()
            start, end = int(start), int(end)
            tup = (source, start, end, feedback)
            data.append(tup)

    return data


def detruecase(lst):
    if lst[0][0].islower():
        lst[0] = lst[0][0].upper() + lst[0][1:]
    return lst


def make_consequtive_edit_array(edits):
    arr = np.zeros((len(edits), len(edits)), dtype = int)

    for x in range(len(edits)):
        arr[x][x] = 1

        for y in range(len(edits))[:x][::-1]:
            if edits[y].end == edits[y + 1].start:
                arr[x][y] = 1
            else:
                break

        for y in range(len(edits))[x + 1:]:
            if edits[y - 1].end == edits[y].start:
                arr[x][y] = 1
            else:
                break

    return arr


def filter_edits(edits, start, end):
    arr = make_consequtive_edit_array(edits)

    new_edits = []
    for x in range(len(edits)):
        indices = [i for i in range(len(edits)) if arr[x, i] == 1]
        if all((edits[i].end < start) or (end < edits[i].start) for i in indices):
            new_edits.append(edits[x])

    return new_edits


def apply_edits(source, start, end, edits):
    offset = 0
    target = source.copy()
    for edit in edits:
        if (edit.end < start) or (end < edit.start):
            target[edit.start + offset : edit.end + offset] = edit.cor
            diff = len(edit.cor) - (edit.end - edit.start)
            offset += diff
            if edit.end < start:
                start += diff
                end += diff
    return target, start, end


def main():
    args = parse_args()

    m2 = load_m2(args.m2)
    tsv = load_tsv(args.tsv)

    for pair, (source, start, end, feedback) in zip(m2, tsv):
        assert ' '.join(pair.src) == ' '.join(source)
        offset = 0

        target = source.copy()
        filtered_edits = filter_edits(pair.edits, start, end)

        target, start, end = apply_edits(source, start, end, filtered_edits)
        target = detruecase(target)

        if feedback is None:
            output = '{}\t{}\t{}'.format(' '.join(target), start, end)
        else:
            output = '{}\t{}\t{}\t{}'.format(' '.join(target), start, end, feedback)
        print(output)

