class Batch:

    def __init__(
            self,
            lengths = None,
            misc = None):

        self.lengths = lengths
        self.misc = misc

    def __len__(self):
        return len(self.lengths)

    def get_num_tokens(self):
        return sum(self.lengths)


class BARTBatch(Batch):

    def __init__(
            self,
            encoder_inputs,
            encoder_attention_mask = None,
            decoder_inputs = None,
            decoder_outputs = None,
            decoder_attention_mask = None,
            lengths = None,
            misc = None):

        super().__init__(
                lengths = lengths,
                misc = misc)

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs

        self.encoder_attention_mask = encoder_attention_mask
        self.decoder_attention_mask = decoder_attention_mask

    def cuda(self):
        self.encoder_inputs = self.encoder_inputs.cuda()

        if self.decoder_inputs is not None:
            self.decoder_inputs = self.decoder_inputs.cuda()

        if self.decoder_outputs is not None:
            self.decoder_outputs = self.decoder_outputs.cuda()

        if self.encoder_attention_mask is not None:
            self.encoder_attention_mask = self.encoder_attention_mask.cuda()

        if self.decoder_attention_mask is not None:
            self.decoder_attention_mask = self.decoder_attention_mask.cuda()

        return self


class GPT2Batch(Batch):

    def __init__(
            self,
            inputs,
            outputs = None,
            attention_mask = None,
            lengths = None,
            misc = None):

        super().__init__(
                lengths = lengths,
                misc = misc)

        self.inputs = inputs
        self.outputs = outputs
        self.attention_mask = attention_mask

    def cuda(self):
        self.inputs = self.inputs.cuda()

        if self.outputs is not None:
            self.outputs = self.outputs.cuda()

        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.cuda()

        return self

