# Multi-Modal Perceiver

## Dataset
Image captioning:
- Flickr8K 8,092 images, each with 5 different captions
- Flickr30K 31,000 images, each with 5 different annotations
- IAPR TC-12 20,000 images, with text caption in 3 languages (English, German, and Spanish)
- MS-COCO over 82,000 images, with at least 5 different captions see: https://cocodataset.org/

Text-to-image generation:
- CUB 200 birds species
- Oxford 102 102 flower categories
- Multi-Modal-CelebA-HQ 30,000 face images see: https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset
- GeNeVA (CoDraw and i-CLEVR) see: https://github.com/Maluuba/GeNeVA

Image inpainting/completion:
- Places 10+millions from 400+ unique scene categories http://places2.csail.mit.edu/index.html

Audio-to-Image/Image-to-Audio
- (TODO: Top)

## Useful repositories
- https://github.com/lucidrains/perceiver-pytorch
- https://github.com/idealwhite/TDANet

## Required packages/Installation
- pytorch
- tensorboard

## Reference
- Perceiver: General Perception with Iterative Attention: https://arxiv.org/pdf/2103.03206.pdf
- Attention is All You Need: https://arxiv.org/pdf/1706.03762.pdf
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention: https://arxiv.org/pdf/1502.03044.pdf
- AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE: https://arxiv.org/pdf/2010.11929.pdf
- Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks: https://arxiv.org/pdf/1810.00825.pdf
- Language Models are Unsupervised Multitask Learners: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent: https://arxiv.org/abs/1902.06720
- Translating Math Formula Images to LaTeX Sequences Using Deep
Neural Networks with Sequence-level Training: https://arxiv.org/pdf/1908.11415.pdf
- UNITER: UNiversal Image-TExt
Representation Learning: https://arxiv.org/pdf/1909.11740.pdf
- Text-Guided Neural Image Inpainting: https://arxiv.org/pdf/2004.03212.pdf
- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size: https://arxiv.org/abs/1602.07360
- Learn, Imagine and Create: Text-to-Image Generation from Prior Knowledge: https://papers.nips.cc/paper/2019/file/d18f655c3fce66ca401d5f38b48c89af-Paper.pdf
- Cross-Modal Retrieval Augmentation for Multi-Modal Classification: https://arxiv.org/abs/2104.08108
- https://www.infoq.com/news/2021/04/perceiver-neural-network-model/
- ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING: https://arxiv.org/pdf/2104.09864.pdf
