# Copyright (c) The RationAI team.

from torch.optim.lr_scheduler import LRScheduler


class ListLR(LRScheduler):
    """Sets the LR based on the items in the input list.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_list (int): List of learning rates to be used consecutively in epochs.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_list, last_epoch=-1):
        self.lr_list = lr_list
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= len(self.lr_list):
            return [self.lr_list[-1] for _ in self.optimizer.param_groups]
        return [self.lr_list[self.last_epoch] for _ in self.optimizer.param_groups]
