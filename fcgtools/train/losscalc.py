class LossCalc:

    def __init__(self, trainer):
        self.trainer = trainer


class BARTLossCalc(LossCalc):

    def __call__(self, batch):
        batch.cuda()
        pred = self.trainer.model(
                input_ids = batch.encoder_inputs,
                decoder_input_ids = batch.decoder_inputs,
                attention_mask = batch.encoder_attention_mask,
                decoder_attention_mask = batch.decoder_attention_mask)
        pred = pred.logits
        pred = pred.view(-1, pred.size(-1))
        target = batch.decoder_outputs.view(-1)
        loss = self.trainer.opter.calc_loss(pred, target)
        return loss


class GPT2LossCalc(LossCalc):

    def __call__(self, batch):
        batch.cuda()
        pred = self.trainer.model(
                input_ids = batch.inputs,
                attention_mask = batch.attention_mask)
        pred = pred.logits
        pred = pred.view(-1, pred.size(-1))
        target = batch.outputs.view(-1)
        loss = self.trainer.opter.calc_loss(pred, target)
        return loss

