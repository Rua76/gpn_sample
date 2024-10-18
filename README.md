# GPN sample run code
This is a sample code to show how to load and use pretrained gpn model to generate prediction on input sequence, and how to extract embeddings

It assume you are working on midway3

## 1. install gpn on your working environment with pip
```
pip install git+https://github.com/songlab-cal/gpn.git
```
## 2. clone this repo to your working directory
```

```
## 3. install *git-lfs* and clone their pretrained gpn model from huggingface
suppose you are using mamba
```
mamba install git-lfs
git clone https://huggingface.co/songlab/gpn-brassicales
```
## 4. run the script with sbatch
```
sbatch submit.sbatch
```
