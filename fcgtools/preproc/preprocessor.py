import random as rd
from .data import MemmapDataWriter
from sacremoses import MosesDetokenizer

from fcgtools.util.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def parse_line(line):
    tup = line.strip().split('\t')

    if len(tup) == 3:
        source, word_start, word_end = tup
        feedback = None
    elif len(tup) == 4:
        source, word_start, word_end, feedback = tup
    else:
        assert False

    word_start, word_end = int(word_start), int(word_end)

    return source, word_start, word_end, feedback


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


def cap_span(sent, word_start, word_end, left, right):
    tokens = sent.split()

    sent_left = ' '.join(tokens[:word_start])
    sent_center = left + ' '.join(tokens[word_start : word_end]) + right
    sent_right = ' '.join(tokens[word_end:])

    sent = ' '.join([sent_left, sent_center, sent_right])
    sent = ' '.join(sent.strip().split())
    return sent


def write_source_data(name, corpus):
    with MemmapDataWriter('{}/src'.format(name)) as f:
        for source, _ in corpus:
            f.write(source)
    logger.info('write: {}/src'.format(name))


def write_target_data(name, corpus):
    with MemmapDataWriter('{}/trg'.format(name)) as g:
        for _, feedback in corpus:
            g.write(feedback)
    logger.info('write: {}/trg'.format(name))


def write_source_raw(convert_to_tokens, name, corpus):
    with open('{}/src.txt'.format(name), 'w') as f:
        for source, _ in corpus:
            print(convert_to_tokens(source), file = f)
    logger.info('write (raw): {}/src.txt'.format(name))


def write_target_raw(convert_to_tokens, name, corpus):
    with open('{}/trg.txt'.format(name), 'w') as g:
        for _, feedback in corpus:
            print(convert_to_tokens(feedback), file = g)
    logger.info('write (raw): {}/trg.txt'.format(name))


class Preprocessor:

    def __init__(
            self,
            left,
            right,
            tokenizer,
            add_gector_tag = False,
            gector_tag_sep = None,
            source_detokenize = False,
            target_with_initial_space = False):

        self.left = left
        self.right = right
        self.tokenizer = tokenizer
        self.detokenizer = MosesDetokenizer(lang = 'en')
        self.add_gector_tag = add_gector_tag
        self.gector_tag_sep = gector_tag_sep
        self.source_detokenize = source_detokenize
        self.target_with_initial_space = target_with_initial_space

    def convert_to_tokens(self, ids):
        return ' '.join([self.tokenizer.tokenizer._convert_id_to_token(x) for x in ids])

    def source_to_ids(self, source, word_start, word_end, tag = None):
        source = cap_span(source, word_start, word_end, self.left, self.right)

        if self.source_detokenize:
            source = self.detokenizer.detokenize(source.split())

        if self.add_gector_tag:
            source = source + self.gector_tag_sep + tag

        source_ids = self.tokenizer.tokenize(source)
        return source_ids

    def feedback_to_ids(self, feedback):
        if feedback is None:
            feedback_ids = None
        else:
            if self.target_with_initial_space:
                feedback = ' ' + feedback
            feedback_ids = self.tokenizer.tokenize(feedback)
        return feedback_ids

    def preproc_line(self, line, tag = None):
        source, word_start, word_end, feedback = parse_line(line)
        source_ids = self.source_to_ids(source, word_start, word_end, tag)
        feedback_ids = self.feedback_to_ids(feedback)
        return source_ids, feedback_ids

    def write(self, name, corpus, raw, only_source):
        write_source_data(name, corpus)
        if not only_source:
            write_target_data(name, corpus)
        if raw:
            write_source_raw(self.convert_to_tokens, name, corpus)
            if not only_source:
                write_target_raw(self.convert_to_tokens, name, corpus)

    def __call__(
            self,
            corpus_path,
            name,
            size = None,
            raw = False,
            only_source = False,
            tag_path = None):

        logger.info('start preprocess')

        corpus = load_corpus(corpus_path, size)

        if self.add_gector_tag:
            with open(tag_path) as f:
                tag_list = [x.strip() for x in f]
            corpus = [self.preproc_line(line, tag) for line, tag in zip(corpus, tag_list)]
        else:
            corpus = [self.preproc_line(line) for line in corpus]

        self.write(name, corpus, raw, only_source)

