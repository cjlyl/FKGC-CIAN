# CIAN

This repo shows the source code of EMNLP 2022 paper: **Learning Inter-Entity Interaction for Few-Shot Knowledge Graph Completion**. In this work, we propose a Cross Interaction Attention Network (CIAN) for few-shot knowledge graph completion.

## Running the Experiments

### Requirements

+ Python 3.6.7
+ PyTorch 1.10.0
+ cuda 11.1



### Dataset

We use NELL-One and Wiki-One to test our MetaR, and these datasets were firstly proposed by xiong. The orginal datasets and pretrain embeddings can be downloaded from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). You can also download the zip files where we put the datasets and pretrain embeddings together from [Dropbox](https://www.dropbox.com/sh/d04wbxx8g97g1rb/AABDZc-2pagoGhKzNvw0bG07a?dl=0). Note that all these files were provided by xiong and we just select what we need here.

### How to run
