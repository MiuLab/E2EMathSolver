# End-to-End Math Solver

The implementation of the NAACL 2019 paper [Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems](https://arxiv.org/abs/1811.00720).

## Usage

0. Download required files:
   - Math23K dataset. If it is not in the valid JSON format, you need to fix it.
   - [Chinese FastText word vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz)
1. Preprocess dataset:

```
cd src/
mkdir ../data/
python make_train_valid.py ~/storage/MWP/data/math23k_fix.json ~/storage/cc.zh.300.vec ../data/train_valid.pkl --valid_ratio 0 --char_based
```
Note that the warning is not unexpected, because some of the problems use operator out of `+, -, *, /`. The purpose of the flags are as follow:
   - `--valid-ratio 0`: Set the ratio the validation dataset should take. It should be set to 0 when running 5-fold cross validation.
   - `--char_based`: Specify if you want to tokenize the problem text into characters rather into words.
2. Start 5-fold cross-validation by
```
mkdir ../models
python train.py ../data/train_valid.pkl ../models/model.pkl --five_fold
```
