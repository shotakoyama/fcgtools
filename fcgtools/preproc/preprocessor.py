import random as rd
from .data import MemmapDataWriter
from sacremoses import MosesDetokenizer

from fcgtools.util.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def parse_line(line):
    source, word_start, word_end, feedback = line.strip().split('\t')
    word_start, word_end = int(word_start), int(word_end)
    return source, word_start, word_end, feedback


def parse_test_line(line):
    source, word_start, word_end = line.strip().split('\t')
    word_start, word_end = int(word_start), int(word_end)
    return source, word_start, word_end


def filter_by_size(lst, size):
    indices = [n for n in range(len(lst))]
    rd.shuffle(indices)
    indices = indices[:size]
    indices.sort()

    new_lst = []
    for index in indices:
        new_lst.append(lst[index])
    logger.info('filtered')

    return new_lst


def load_corpus(corpus_path, size):

    with open(corpus_path) as f:
        corpus = [x.strip() for x in f]
        if size is not None:
            corpus = filter_by_size(corpus, size)
    logger.info('loaded: {} ({})'.format(corpus_path, len(corpus)))

    return corpus


class Preprocessor:

    def __init__(
            self,
            left,
            right,
            tokenizer,
            source_detokenize,
            target_with_initial_space):

        self.left = left
        self.right = right
        self.tokenizer = tokenizer
        self.detokenizer = MosesDetokenizer(lang = 'en')
        self.source_detokenize = source_detokenize
        self.target_with_initial_space = target_with_initial_space

    def cap_source(self, source, word_start, word_end):
        tokens = source.split()
        source_left = ' '.join(tokens[:word_start])
        source_center = self.left + ' '.join(tokens[word_start : word_end]) + self.right
        source_right = ' '.join(tokens[word_end:])
        source = ' '.join([source_left, source_center, source_right])
        return source

    def convert_to_tokens(self, ids):
        return ' '.join([self.tokenizer.tokenizer._convert_id_to_token(x) for x in ids])

    def preproc_line(self, line, only_source = False):

        if only_source:
            source, word_start, word_end = parse_test_line(line)
        else:
            source, word_start, word_end, feedback = parse_line(line)

        source = self.cap_source(source, word_start, word_end)
        if self.source_detokenize:
            source = self.detokenizer.detokenize(source.split())
        source_ids = self.tokenizer.tokenize(source)

        if only_source:
            return source_ids, None
        else:
            if self.target_with_initial_space:
                feedback = ' ' + feedback
            feedback_ids = self.tokenizer.tokenize(feedback)
            return source_ids, feedback_ids

    def write_data(self, name, corpus, only_source = False):

        with MemmapDataWriter('{}/src'.format(name)) as f:
            for source, _ in corpus:
                f.write(source)
        logger.info('write: {}/src'.format(name))

        if not only_source:
            with MemmapDataWriter('{}/trg'.format(name)) as g:
                for _, feedback in corpus:
                    g.write(feedback)
            logger.info('write: {}/trg'.format(name))

    def write_raw(self, name, corpus, only_source = False):

        with open('{}/src.txt'.format(name), 'w') as f:
            for source, _ in corpus:
                print(self.convert_to_tokens(source), file = f)
        logger.info('write (raw): {}/src.txt'.format(name))

        if not only_source:
            with open('{}/trg.txt'.format(name), 'w') as g:
                for _, feedback in corpus:
                    print(self.convert_to_tokens(feedback), file = g)

            logger.info('write (raw): {}/trg.txt'.format(name))

    def __call__(self, corpus_path, name, size = None, raw = False, only_source = False):
        logger.info('start preprocess')

        corpus = load_corpus(corpus_path, size)
        corpus = [self.preproc_line(line, only_source) for line in corpus]

        self.write_data(name, corpus, only_source)

        if raw:
            self.write_raw(name, corpus, only_source)

