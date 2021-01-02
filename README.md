# Subformer

This repository contains the code for the Subformer. To help overcome this we propose the Subformer, allowing us to retain performance while reducing parameters in generative Transformers from 25% ~ 70%. The Subformer consists of the following two techniques:

1. Sandwich-style parameter sharing, in which we share all the layers in a block except the first and last. This allows us the use the central shared layers --"sandwich module" -- as a large representation learner (similar to BERT vs ALBERT) while the input and output model layers are able to focus on more specific representations for token prediction/generation while maintaining performance.
2. For our sequence to sequence tasks, we also introduce SAFE (self-attentive factorized embeddings), which help us reduce embedding parameters significantly, while still retaining performance.

If you used this code or found our work useful, please cite:

```bibtex
@misc{reid2021subformer,
  title = {{S}ubformer: {E}xploring {W}eight {S}haring for {P}arameter {E}fficiency in {G}enerative {T}ransformers},
  author = {Machel Reid and Edison Marrese-Taylor and Yutaka Matsuo},
  year = {2021},
  url = {https://machelreid.github.io/resources/subformer.pdf}
}
```

# Requirements and Installation

(As this code is based on [fairseq](https://github.com/ytorch/fairseq/), some installation instructions are taken straight from their README)

* PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* To install and develop locally:

``` bash
git clone https://github.com/machelreid/subformer
cd subformer
pip install --e ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Training

```bash
python train.py $DATA_BIN --arch transformer_wmt_en_de \
    --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 \
    --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation \
    --max-tokens 8192 --weight-decay 0.01 --dropout 0.2 --encoder-layers 6 --encoder-embed-dim 512 \
    --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 \
    --max-target-positions 10000 --max-update 200000 --seed 1 \
    --save-dir $CHECKPOINT_DIR --share-all-embeddings \
    --share-encoder-parameters-sandwich --share-decoder-parameters-sandwich \ #for sandwich-style parameter sharing
    --reduction-dim 320 #for SAFE embeddings
```

# Evaluation

```bash
python generate.py --path $CHECKPOINT --gen-subset $SPLIT --beam 5 --lenpen 0.2 --batch-size 400 --remove-bpe
```

# Citation

Please cite as:

``` bibtex
@misc{reid2021subformer,
  title = {{S}ubformer: {E}xploring {W}eight {S}haring for {P}arameter {E}fficiency in {G}enerative {T}ransformers},
  author = {Machel Reid and Edison Marrese-Taylor and Yutaka Matsuo},
  year = {2021},
  url = {https://machelreid.github.io/resources/subformer.pdf}
}
```
