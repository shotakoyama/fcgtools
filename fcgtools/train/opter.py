import torch
import torch.nn as nn
import torch.optim as optim

class Opter:

    def __init__(
            self,
            model,
            lr,
            max_grad_norm,
            scheduler = 'constant',
            warmup_steps = 0,
            start_factor = 1.0/3,
            weight_decay = 0.01):

        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index = -100)
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_grad_norm = max_grad_norm
        self.total_grad_norm = None

        self.optimizer = optim.AdamW(
                model.parameters(),
                lr = lr,
                weight_decay = weight_decay)

        if scheduler == 'constant':
            self.scheduler = optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    total_iters = warmup_steps,
                    factor = start_factor)
        elif scheduler == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    total_iters = warmup_steps,
                    start_factor = start_factor)
        else:
            assert False

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def calc_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss

    def step(self):
        self.scaler.unscale_(self.optimizer)
        self.total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

