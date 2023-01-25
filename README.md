# sdsim

stable diffusion model similarity calculatior

original: <https://huggingface.co/JosephusCheung/ASimilarityCalculatior>

modified: <https://gist.github.com/toriato/87db2b3cbe9863be43b63ba10cd610ad>

packaging: here


## Install

```sh
pip install sdsim
```

## Usage

```sh
sdsim animefull-latest.ckpt ACertainModel.safetensors
```

```sh
sdsim -nv -j result.json animefull-latest.ckpt ACertainModel.safetensors
```

```sh
❯ sdsim --help
usage: sdsim [-h] [-j JSON] [-s SEED] [-nv] base targets [targets ...]

positional arguments:
  base
  targets

optional arguments:
  -h, --help            show this help message and exit
  -j JSON, --json JSON
  -s SEED, --seed SEED
  -nv, --no-verbose
```

```python
from sdsim import similarity

model1 = "animefull-latest.ckpt"
model2 = "ACertainModel.safetensors"
model3 = "AbyssOrangeMix2_nsfw.safetensors"
result = similarity(model1, [model2, model3], verbose=True)
```
