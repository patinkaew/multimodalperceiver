import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_num_mode_parameters(model):
    return sum(p.numel() for p in model.parameters())

def show_image(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def strip_start_end(caption):
    text = caption.strip()
    if text.startswith("<START>"):
        text = text[len("<START>"):]
    if text.endswith("<END>"):
        text = text[:-len("<END>")]
    return text.strip()

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded
