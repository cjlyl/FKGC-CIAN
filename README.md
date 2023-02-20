# CIAN

This repo shows the source code of EMNLP 2022 paper: **Learning Inter-Entity Interaction for Few-Shot Knowledge Graph Completion**. In this work, we propose a Cross Interaction Attention Network (CIAN) for few-shot knowledge graph completion.

## Running the Experiments

### Requirements

+ Python 3.6.7
+ PyTorch 1.10.0
+ cuda 11.1
+ GPU 3090



### Dataset

We use NELL-One and Wiki-One to test our MetaR, and these datasets were firstly proposed by xiong. The orginal datasets and pretrain embeddings can be downloaded from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). You can also download the zip files where we put the datasets and pretrain embeddings together from [Dropbox](https://www.dropbox.com/sh/d04wbxx8g97g1rb/AABDZc-2pagoGhKzNvw0bG07a?dl=0). Note that all these files were provided by xiong and we just select what we need here.


### How to run

####NELL-One

```bash
# NELL-One, 5-shot,
python main.py --fine_tune --lr 8e-5 --few 5 --prefix nelllr8e-5.5shot```
```

```bash
# NELL-One, 3-shot,
python main.py --fine_tune --lr 8e-5 --few 3 --prefix nelllr8e-5.3shot```
```

####Wiki-One

```bash
# Wiki-One, 5-shot,
python main.py --fine_tune --lr 2e-4 --few 5 --prefix wikilr2e-4.5shot```
```

```bash
# Wiki-One, 3-shot,
python main.py --fine_tune --lr 2e-4 --few 3 --prefix wikilr2e-4.3shot```
```


Here are explanations of some important args,

```bash
--data_path: "directory of dataset"
--few:       "the number of few in {few}-shot, as well as instance number in support set"
--prefix:    "given name of current experiment"
--fine_tune  "whether to fine tune the pre_trained embeddings"
--device:    "the GPU number"
```

%Normally, other args can be set to default values. See ``params.py`` for more details about argus if needed.



