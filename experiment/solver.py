import numpy as np
import torch

class BaseSolver:
    def __init__(self, model, optimizer):
        self.model = model
        self.optim = optimizer
        self._reset()

    def _reset(self):
        #self.epoch = 0
        self.loss_history = []

class ClassifierSolver(BaseSolver):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def train(loader_train,
              num_epochs = 1,
              loader_val = None,
              device = torch.device("cpu")):

        self.model.train()
        for e in range(start_epoch, start_epoch + num_epochs):
            for i, (images, labels) in enumerate(loader_train):
                pass


class CaptioningSolver(BaseSolver): # simplified solver from assign3
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def train(loader_train,
              num_epochs = 1,
              loader_val = None,
              device = torch.device("cpu"),
              start_epoch = 0):

        self.model.train()
        for e in range(start_epoch, start_epoch + num_epochs):
            for i, (images, captions) in enumerate(loader_train):
                t_images = images.to(device)
                t_captions_in = captions[:, :-1].to(device)
                t_captions_out = captions[:, 1:].to(device)
                t_mask = (t_captions_out != self.model._null).to(device)

                logits = self.model(t_features, t_captions_in)
                loss = self._temporal_softmax_loss(logits, t_captions_out, t_mask)
                self.loss_history.append(loss.detach().numpy())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def _temporal_softmax_loss(self, logits, t_captions_out, t_mask):
        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction= "none")
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
