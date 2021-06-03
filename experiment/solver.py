import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from model.utils import strip_start_end, decode_captions

class CaptioningSolver: # simplified solver from assign3
    def __init__(self, encoder, decoder,
                encoder_optimizer, decoder_optimizer,
                idx_to_word, caption_alias,
                scheduler = None,
                print_every=10, verbose=True,
                device = torch.device("cpu")):
        self.encoder = encoder.to(device=device, dtype=torch.float32)
        self.decoder = decoder.to(device=device, dtype=torch.float32)
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.scheduler = scheduler
        self.print_every = print_every
        self.verbose = verbose
        self.device = device
        self.idx_to_word = idx_to_word
        self.caption_alias = caption_alias
        self._reset()

    def _reset(self):
        # Set up some variables for book-keeping
        self.global_step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_bleu = []

    def _train_step(self, loader_train): # train for one epoch
        self.encoder.train()
        self.decoder.train()
        num_batches = len(loader_train)
        for t, (images, captions) in enumerate(loader_train):
            if self.verbose:
                if self.global_step % self.print_every == 0:
                    print("training batch {} of {} batches".format(t, num_batches))
            # move to device
            images = images.to(device=self.device, dtype=torch.float32)
            captions = captions.to(device=self.device, dtype=torch.long)
            # compute image features
            features = self.encoder(images)

            # prepare captions
            captions_in = captions[:, :-1]
            captions_out = captions[:, 1:]
            mask = captions_out != self.decoder._null

            # produce caption scores
            scores = self.decoder(features, captions_in)

            loss = self.temporal_softmax_loss(scores, captions_out, mask)
            self.train_loss_history.append(loss.detach().cpu().numpy())
            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()
            self.global_step += 1

    def _val_step(self, loader_val):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            reference = []
            hypothesis = []
            for t, (images, captions) in enumerate(loader_val):
                # move to device
                images = images.to(device=self.device, dtype=torch.float32)
                captions = captions.to(device=self.device, dtype=torch.long)
                # compute image features
                features = self.encoder(images)

                # prepare captions
                captions_in = captions[:, :-1]
                captions_out = captions[:, 1:]
                mask = captions_out != self.decoder._null

                # produce caption scores
                scores = self.decoder(features, captions_in)

                loss = self.temporal_softmax_loss(scores, captions_out, mask)
                self.val_loss_history.append(loss.detach().cpu().numpy())

                decode_refs = []
                for caption in captions:
                    decode_ref = decode_captions(caption.detach().cpu().numpy(), self.idx_to_word)
                    decode_refs.append(self.caption_alias[decode_ref])
                reference.append(decode_refs)

                caption_pred = torch.argmax(scores, dim=2)
                decode_pred = decode_captions(caption_pred.detach().cpu().numpy(), self.idx_to_word)
                hypothesis.append(decode_pred)

            assert len(reference) == len(hypothesis)
            if self.verbose:
                print("computing BLEU score on validation")
            bleu = corpus_bleu(reference, hypothesis)
            self.val_bleu.append(bleu)

    def train(self,
              loader_train,
              loader_val = None,
              num_epochs = 1,
              start_epoch = 0):
        for e in range(start_epoch, start_epoch + num_epochs):
            if self.verbose:
                print("begin training epoch {} of {} epochs".format(e+1, num_epochs))
            self._train_step(loader_train)
            if self.verbose:
                print("finish training epoch {} of {} epochs".format(e+1, num_epochs))

            if loader_val is not None:
                if self.verbose:
                    print("begin validation epoch {} of {} epochs".format(e+1, num_epochs))
                self._val_step(loader_val)
                if self.verbose:
                    print("finish validation epoch {} of {} epochs".format(e+1, num_epochs))
            if self.scheduler is not None:
                if self.verbose:
                    print("adjusting LR scheduler after epoch {} of {} epochs".format(e+1, num_epochs))
                self.scheduler.step()

    def temporal_softmax_loss(self, x, y, mask):
        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
