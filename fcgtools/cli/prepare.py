from argparse import ArgumentParser
import re

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--test')
    return parser.parse_args()


def char_span_to_pair(span):
    char_start, char_end = span.split(':')
    char_start, char_end = int(char_start), int(char_end)
    return char_start, char_end


def load_data(path):
    with open(path) as f:
        for x in f:
            tup = x.strip().split('\t')

            if len(tup) == 3:
                source, span, feedback = tup
            elif len(tup) == 2:
                source, span = tup
                feedback = None
            else:
                assert False

            char_start, char_end = char_span_to_pair(span)
            yield source, char_start, char_end, feedback


def output_data(data, path):
    with open(path, 'w') as f:
        for source, word_start, word_end, feedback in data:

            if feedback is None:
                line = '{}\t{}\t{}'.format(
                        source, word_start, word_end)
            else:
                line = '{}\t{}\t{}\t{}'.format(
                        source, word_start, word_end, feedback)

            print(line, file = f)


def fix_train(data):
    for index, (source, char_start, char_end, feedback) in enumerate(data, start = 1):

        if index == 782:
            assert char_start == 40
            assert char_end == 64
            char_start = 41

        if index == 860:
            assert char_start == 52
            assert char_end == 61
            char_end = 59

        if index == 1995:
            assert char_start == 73
            assert char_end == 76
            char_end = 75

        if index == 3775:
            assert char_start == 164
            assert char_end == 175
            char_start = 165

        if index in {23, 330, 1773, 2614, 2864, 3120, 3158, 3416, 3503, 3580, 4090, 4267, 4304, 4411, 4412, 4413}:
            assert 'precede the what' in feedback
            feedback = feedback.replace('precede the what', 'precede what')

        if index == 104:
            assert ' << agree>> ' in feedback
            feedback = feedback.replace(' << agree>> ', ' <<agree>> ')

        if index in {139, 1128, 1130, 2188, 2189, 4780}:
            assert "''to come up with / remember something'" in feedback
            feedback = feedback.replace(
                    "''to come up with / remember something'",
                    "``to come up with / remember something''")

        if index == 142:
            assert ' <<of >> ' in feedback
            feedback = feedback.replace(' <<of >> ', ' <<of>> ')

        if index in {153, 487, 1493, 1574, 1698, 3542, 4196}:
            assert ' . ' in feedback
            feedback = feedback.replace(' . ', '. ')

        if index in {173, 2991}:
            assert 't<' in feedback
            feedback = feedback.replace('t<', 't <')

        if index == 216:
            assert '’instead' in feedback
            feedback = feedback.replace('’instead', 'instead')

        if index in {289, 319, 445, 483, 1519}:
            assert ' <<in> ' in feedback
            feedback = feedback.replace(' <<in> ', ' <<in>> ')

        if index == 369:
            assert ' ｓ' in feedback
            feedback = feedback.replace(' ｓ', '')

        if index == 579:
            assert '<<Near>>> ' in feedback
            feedback = feedback.replace('<<Near>>> ', '<<Near>> ')

        if index in {793, 3961}:
            assert ' ‘' in feedback
            feedback = feedback.replace(' ‘', ' `')

        if index == 925:
            assert '<object,> use' in feedback
            feedback = feedback.replace('<object,> use', '<object>, use')

        if index in {1054, 2372, 2374, 2549, 2551, 2553, 2648, 2746, 3218, 4445, 4624}:
            assert ' , ' in feedback
            feedback = feedback.replace(' , ', ', ')

        if index == 1069:
            assert ' <<change> ' in feedback
            feedback = feedback.replace(' <<change> ', ' <<change>> ')

        if index == 1071:
            assert " 'of'' " in feedback
            feedback = feedback.replace(" 'of'' ", " ``of'' ")

        if index in {1109, 2409, 4036}:
            assert ' <verb>> ' in feedback
            feedback = feedback.replace(' <verb>> ', ' <verb> ')

        if index == 1321:
            assert 'wed”' in feedback
            feedback = feedback.replace('wed”', "wed''")

        if index in {1404, 1755, 2837, 3886}:
            assert '>m' in feedback
            feedback = feedback.replace('>m', '> m')

        if index == 1748:
            assert "'''" in feedback
            feedback = feedback.replace("'''", '``')

        if index == 1788:
            assert "‘' " in feedback
            feedback = feedback.replace("‘' ", "'' ")

        if index == 2187:
            assert ' <<on> ' in feedback
            feedback = feedback.replace(' <<on> ', ' <<on>> ')

        if index == 2221:
            assert '>f' in feedback
            feedback = feedback.replace('>f', '> f')

        if index == 2556:
            assert "'''" in feedback
            feedback = feedback.replace("'''", "'")

        if index == 2803:
            assert "'''" in feedback
            feedback = feedback.replace("'''", "''")

        if index == 2914:
            assert ' <<between> ' in feedback
            feedback = feedback.replace(' <<between> ', ' <<between>> ')

        if index == 2932:
            assert '>>h' in feedback
            feedback = feedback.replace('>>h', '>>')

        if index == 2950:
            assert "'peripherally in''" in feedback
            feedback = feedback.replace("'peripherally in''", "``peripherally in''")

        if index == 2964:
            assert "''to give an example'" in feedback
            feedback = feedback.replace("''to give an example'", "``to give an example''")

        if index == 2982:
            assert '　' in feedback
            assert ",''" in feedback
            feedback = feedback.replace('　', ' ')
            feedback = feedback.replace(",''", "'',")

        if index == 3087:
            assert ' << some students>> '
            feedback = feedback.replace(' << some students>> ', ' <<some students>> ')

        if index == 3144:
            assert ' <<care> ' in feedback
            feedback = feedback.replace(' <<care> ', ' <<care>> ')

        if index == 3222:
            assert ' << as>> ' in feedback
            feedback = feedback.replace(' << as>> ', ' <<as>> ')

        if index == 3260:
            assert '>c' in feedback
            feedback = feedback.replace('>c', '> c')

        if index == 3265:
            assert ' << college >> ' in feedback
            assert ' ,<<at>> ' in feedback
            feedback = feedback.replace(' << college >> ', ' <<college>> ')
            feedback = feedback.replace(' ,<<at>> ', ', <<at>> ')

        if index == 3277:
            assert ' <preposition>> ' in feedback
            feedback = feedback.replace(' <preposition>> ', ' <preposition> ')

        if index == 3789:
            assert 'e<' in feedback
            feedback = feedback.replace('e<', 'e <')

        if index == 3987:
            assert ' <<worry> ' in feedback
            feedback = feedback.replace(' <<worry> ', ' <<worry>> ')

        if index == 4031:
            assert ' <<talk> ' in feedback
            feedback = feedback.replace(' <<talk> ',  ' <<talk>> ')

        if index == 4032:
            assert '“skill”' in feedback
            feedback = feedback.replace('“skill”', "``skill''")

        if index == 4316:
            assert ' << demand>> ' in feedback
            feedback = feedback.replace(' << demand>> ', ' <<demand>> ')

        if index == 4381:
            assert '."' in feedback
            feedback = feedback.replace('."', '".')

        if index == 4548:
            assert '.Use ' in feedback
            feedback = feedback.replace('.Use ', '. Use ')

        if index == 4626:
            assert '>i' in feedback
            feedback = feedback.replace('>i', '> i')

        if index == 4842:
            assert ' <<of >>. ' in feedback
            feedback = feedback.replace(' <<of >>. ', ' <<of>>. ')

        yield source, char_start, char_end, feedback


def fix_feedback(data):
    for source, char_start, char_end, feedback in data:

        feedback = ' '.join(feedback.split())

        if '’' in feedback:
            feedback = feedback.replace('’', "'")

        if '…' in feedback:
            feedback = feedback.replace('…', '...')

        if '"' in feedback:
            feedback = feedback.replace('"', "''")

        if 'into into' in feedback:
            feedback = feedback.replace('into into', 'into')

        feedback = re.sub(r"(?<!\+) ''", " ``", feedback)
        feedback = re.sub(r"(?<!\+) '", " `", feedback) # not to change "+ 's" to "+ `s"
        feedback = re.sub(r"^''", "``", feedback)
        feedback = re.sub(r"^'", "`", feedback)
        feedback = re.sub(r"(?<!`)`([^`']*)'(?=\W)(?![^`'<]*>)", "``\\1''", feedback)


        yield source, char_start, char_end, feedback


def parse_word_start_end_dict(source):
    word_start_list = []
    word_end_list = []

    tmp = 0
    for i in range(len(source)):
        if source[i] == ' ':
            word_start_list.append(tmp)
            word_end_list.append(i)
            tmp = i + 1
    word_start_list.append(tmp)
    word_end_list.append(i + 1)

    word_start_dict = {x: i for i, x in enumerate(word_start_list)}
    word_end_dict = {x: i + 1 for i, x in enumerate(word_end_list)}

    return word_start_dict, word_end_dict


def parse_word_span(source, char_start, char_end):

    word_start_dict, word_end_dict = parse_word_start_end_dict(source)
    assert char_start in word_start_dict
    assert char_end in word_end_dict

    word_start = word_start_dict[char_start]
    word_end = word_end_dict[char_end]
    return word_start, word_end


def convert_char_span_to_word_span(data):
    for source, char_start, char_end, feedback in data:
        word_start, word_end = parse_word_span(source, char_start, char_end)
        yield source, word_start, word_end, feedback


def convert_source_lrb_rrb(source):
    new_source = []
    for token in source.split():
        if token == '-LRB-':
            new_token = '('
        elif token == '-RRB-':
            new_token = ')'
        else:
            new_token = token
        new_source.append(new_token)
    new_source = ' '.join(new_source)
    return new_source


def fix_lrb_rrb(data):
    for source, word_start, word_end, feedback in data:
        source = convert_source_lrb_rrb(source)
        yield source, word_start, word_end, feedback


def train_main(path):
    data = load_data(path)
    data = fix_train(data)
    data = fix_feedback(data)
    data = convert_char_span_to_word_span(data)
    data = fix_lrb_rrb(data)
    output_data(data, 'train.tsv')


def valid_main(path):
    data = load_data(path)
    data = convert_char_span_to_word_span(data)
    data = fix_lrb_rrb(data)
    output_data(data, 'valid.tsv')


def test_main(path):
    data = load_data(path)
    data = convert_char_span_to_word_span(data)
    data = fix_lrb_rrb(data)
    output_data(data, 'test.tsv')


def main():
    args = parse_args()

    if args.train is not None:
        train_main(args.train)

    if args.valid is not None:
        valid_main(args.valid)

    if args.test is not None:
        test_main(args.test)

