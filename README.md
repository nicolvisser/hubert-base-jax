# HuBERT Base JAX Implementation

An implementation of HuBERT Base in JAX. (I want to be able to pretrain and finetune HuBERT efficiently on TPUs)

This repository is a **work in progress** and is not yet complete.

To Do List:
- [x] Build the model for inference
- [x] Map and import weights from [bshall/hubert:main](https://github.com/bshall/hubert)
- [x] Add padding mask
- [x] Test pretrained model ABX on LibriSpeech
- [x] Add masking strategy
- [x] Build dataset prepare scripts and loader
- [x] Build trainer module
- [x] Test pretraining on LibriSpeech dataset (single GPU)
- [ ] Add LoRA
- [ ] Test LoRA finetuning for phoneme recognition
- [ ] Extend training to multiple TPUs with data parallelism
- [ ] Clean up code and add documentation

This repository is based on the following work:
- Benji van Niekerk's stripped down [implementation](https://github.com/bshall/hubert) of HuBERT Base and easily accessible [weights](https://github.com/bshall/hubert/releases/tag/v0.2).
- Phillip Lippe's [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html) on building a transformer in JAX.
- The HuBERT [paper](https://arxiv.org/abs/2106.07447).
- The fairseq [repo](https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/hubert).

## Installation
Install JAX for your system by following [these](https://jax.readthedocs.io/en/latest/installation.html) instructions. For example, for CUDA 12.0, you can run the following command:

```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

I also use PyTorch on the CPU to use their datasests and dataloaders as well as loading the weights from the PyTorch checkpoint:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install other requirements

```bash
pip install -r requirements.txt
```

Note: If you want to compute the ABX score, I recommend installing and using `zrc_abx2` in a separate environment.

# Results

ABX results for each layer on LibriSpeech when using the weights from [bshall/hubert:main](https://github.com/bshall/hubert).

| Layer Index | ABX<br>Within<br>Within | ABX<br>Any<br>Within | ABX<br>Within<br>Across | ABX<br>Any<br>Across |
| ----------- | ----------------------- | -------------------- | ----------------------- | -------------------- |
| 0           | 6.15                    |                      |                         |                      |
| 1           | 6.03                    |                      |                         |                      |
| 2           | 5.13                    |                      |                         |                      |
| 3           | 4.20                    |                      |                         |                      |
| 4           | 3.41                    |                      |                         |                      |
| 5           | 2.77                    | 10.04                | 3.67                    | 10.74                |
| 6           | 2.38                    | 9.75                 | 3.10                    | 10.33                |
| 7           | 2.32                    | 10.24                | 3.20                    | 10.81                |
| 8           | 2.39                    | 10.23                | 3.16                    | 10.76                |
| 9           | 1.97                    | 8.77                 | 2.74                    | 9.20                 |
| **10**      | **1.91**                | **8.52**             | **2.60**                | **9.12**             |
| 11          | 2.12                    | 8.79                 | 2.94                    | 9.34                 |


