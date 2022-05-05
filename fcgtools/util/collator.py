import random as rd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from .batch import (
        BARTBatch,
        GPT2Batch)

class Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.bos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.eos = self.tokenizer.eos_token_id

    def pad_outputs(self, outputs):
        return pad(
                outputs,
                padding_value = -100,
                batch_first = True)

    def make_lengths(self, batch):
        return [
                len(dct['source'])
                for dct
                in batch]


class BARTCollator(Collator):

    def pad_inputs(self, inputs):
        return pad(
                inputs,
                padding_value = self.pad,
                batch_first = True)

    # BART Decoder starts with eos token and bos token.
    # Not only with bos token, although it would be right.
    # https://github.com/huggingface/transformers/issues/3668

    def make_encoder_inputs(self, batch):
        return [
            torch.tensor(
                [self.bos] + dct['source'] + [self.eos])
            for dct
            in batch]

    def make_decoder_inputs(self, batch):
        return [
            torch.tensor(
                [self.eos, self.bos] + dct['target'])
            for dct
            in batch]

    def make_decoder_outputs(self, batch):
        return [
            torch.tensor(
                [self.bos] + dct['target'] + [self.eos])
            for dct
            in batch]

    def __call__(self, batch):
        ei = self.pad_inputs(self.make_encoder_inputs(batch))
        di = self.pad_inputs(self.make_decoder_inputs(batch))
        do = self.pad_outputs(self.make_decoder_outputs(batch))
        lengths = self.make_lengths(batch)

        eam = ei.ne(self.pad).long()
        dam = di.ne(self.pad).long()

        batch = BARTBatch(
                encoder_inputs = ei,
                encoder_attention_mask = eam,
                decoder_inputs = di,
                decoder_outputs = do,
                decoder_attention_mask = dam,
                lengths = lengths)
        return batch


class GPT2Collator(Collator):

    def __init__(
            self,
            tokenizer,
            sep,
            backward_only_target = False):

        super().__init__(tokenizer)
        self.sep = self.tokenizer.encode(sep)

        if backward_only_target:
            self.make_outputs = self.make_half_outputs
        else:
            self.make_outputs = self.make_full_outputs

    def pad_inputs(self, inputs):
        return pad(
                inputs,
                padding_value = self.eos,
                batch_first = True)

    def make_inputs(self, batch):
        return [
            torch.tensor(
                [self.bos] + dct['source']
                + self.sep + dct['target'])
            for dct
            in batch]

    def make_full_outputs(self, batch):
        return [
            torch.tensor(
                dct['source'] + self.sep
                + dct['target'] + [self.eos])
            for dct
            in batch]

    def make_half_outputs(self, batch):
        return [
            torch.tensor(
                ([-100] * (len(dct['source']) + len(self.sep)))
                + dct['target'] + [self.eos])
            for dct
            in batch]

    def __call__(self, batch):
        inputs = self.pad_inputs(self.make_inputs(batch))
        outputs = self.pad_outputs(self.make_outputs(batch))
        lengths = self.make_lengths(batch)

        attn_mask = inputs.ne(self.eos).long()
        return GPT2Batch(
                inputs = inputs,
                outputs = outputs,
                attention_mask = attn_mask,
                lengths = lengths)

