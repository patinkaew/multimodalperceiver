# adapt from https://www.kaggle.com/mdteach/torch-data-loader-flicker-8k
from collections import Counter, defaultdict
import numpy as np
import spacy
import torch
import pandas as pd
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
from model.utils import decode_captions

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<NULL>",1:"<START>",2:"<END>",3:"<UNK>"}

        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]

class CaptionAlias:
    def __init__(self, caption_df, vocab):
        self.img_to_captions = defaultdict(list)
        self.caption_to_img = dict()
        for img, caption in caption_df.to_numpy():
            caption_vec = []
            caption_vec += [vocab.stoi["<START>"]]
            caption_vec += vocab.numericalize(caption)
            caption_vec += [vocab.stoi["<END>"]]
            decode = decode_captions(np.array(caption_vec), vocab.itos)
            self.img_to_captions[img].append(decode)
            self.caption_to_img[decode] = img

        self.caption_alias = dict()
        for img, captions in self.img_to_captions.items():
            for caption in captions:
                self.caption_alias[caption] = captions

    def __call__(self, caption):
        return self.caption_alias[caption]

    def __getitem__(self, caption):
        return self.caption_alias[caption]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,caption_file,transform=None,freq_threshold=5, verbose=False):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform

        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Initialize vocabulary and build vocab
        if verbose:
            print("building vocab")
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

        #Initialize caption alias for BLEU computation
        if verbose:
            print("buidling caption alias")
        self.caption_alias = CaptionAlias(self.df, self.vocab)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")

        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<START>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<END>"]]

        return img, torch.tensor(caption_vec)

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets
