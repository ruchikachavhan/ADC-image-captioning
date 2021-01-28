from torch.optim import Adam
import torch


class Optim(object):
    def __init__(self, params, lr, is_pre,
                 grad_clip, new_lr=0.0, weight_decay=0.):
        self.optimizer = Adam(params, lr=lr, betas=(
            0.9, 0.98), eps=1e-09, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.params = params
        self.lr = lr
        if is_pre:
            self.step = self.pre_step
        else:
            assert new_lr != 0.0

            self.n_current_steps = 0
            self.new_lr = new_lr
            self.step = self.train_step

    def train_step(self):
        self.optimizer.step()

        self.n_current_steps += 1
        if (self.n_current_steps%1000 == 0):
            self.update_learning_rate()

    def pre_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm(self.params, self.grad_clip)

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * 0.99

class Policy_optim(Optim):
    def __init__(self, params, lr, grad_clip, new_lr):
        super().__init__(params, lr, False, grad_clip, new_lr)

    def train_step(self, reward):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.grad = p.grad.mul(reward)

        self.optimizer.step()

        self.n_current_steps += 1
        if (self.n_current_steps%2000==0):
            self.update_learning_rate()
