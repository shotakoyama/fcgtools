from sacrebleu import BLEU


scorer = BLEU(effective_order = True)

def fcg_sent_bleu(hyp, ref):
    bleu = scorer.sentence_score(hyp, [ref]).score
    return bleu


def fcg_corpus_score(hyp_list, ref_list):

    assert len(hyp_list) == len(ref_list)

    bleu_list = [
        fcg_sent_bleu(hyp, ref)
        for hyp, ref
        in zip(hyp_list, ref_list)]

    r_list = [
        bleu
        for ref, bleu
        in zip(ref_list, bleu_list)
        if ref != '<NO_COMMENT>']
    recall = sum(r_list) / len(r_list)


    p_list = [
        bleu
        for hyp, bleu
        in zip(hyp_list, bleu_list)
        if hyp != '<NO_COMMENT>']
    precision = sum(p_list) / len(p_list)

    f = (2 * recall * precision) / (recall + precision)
    return f

