from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn
from tqdm.auto import tqdm

ATTN1 = "model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_{qkv}.weight"


@dataclass
class Model:
    path: str | Path

    def __post_init__(self):
        self.hash = model_hash(self.path)
        self.hash_old = model_hash_old(self.path)
        self.state_dict = load_model(self.path)

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path}, hash={self.hash}, hash_old={self.hash_old})"

    def __str__(self):
        return f"{self.path} [{self.hash}; {self.hash_old}]"

    def dict(self):  # noqa: A003
        return {"path": str(self.path), "hash": self.hash, "hash_old": self.hash_old}


def check_path(path: str | Path) -> bool:
    if isinstance(path, str):
        path = Path(path)

    if not path.exists() or path.is_dir():
        return False
    return True


def model_hash_old(file: str | Path) -> str:
    if not check_path(file):
        raise FileNotFoundError(f"File {file} does not exist")

    with open(file, "rb") as f:
        m = hashlib.sha256()

        f.seek(0x100000)
        m.update(f.read(0x10000))
    return m.hexdigest()[:8]


def model_hash(file: str | Path) -> str:
    bulksize = 2**20

    if not check_path(file):
        raise FileNotFoundError(f"File {file} does not exist")

    sha256 = hashlib.sha256()

    with open(file, "rb") as f:
        while True:
            chunk = f.read(bulksize)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()[:10]


def load_model(path: str | Path) -> dict[str, Tensor]:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")

    ckpt = torch.load(path, map_location="cpu")
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def cal_cross_attn(
    to_q: Tensor, to_k: Tensor, to_v: Tensor, rand_input: Tensor
) -> Tensor:
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q})
    attn_to_k.load_state_dict({"weight": to_k})
    attn_to_v.load_state_dict({"weight": to_v})

    qk = torch.einsum("ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input))
    qk = F.softmax(qk, dim=-1)

    return torch.einsum("ik, jk -> ik", qk, attn_to_v(rand_input))


def evaluate(state_dict: dict[str, Tensor], n: int, rand_input: Tensor) -> Tensor:
    qk = ATTN1.format(n=n, qkv="q")
    uk = ATTN1.format(n=n, qkv="k")
    vk = ATTN1.format(n=n, qkv="v")
    atoq, atok, atov = state_dict[qk], state_dict[uk], state_dict[vk]

    attn = cal_cross_attn(atoq, atok, atov, rand_input)
    return attn


def similarity(
    base: str | Path,
    targets: str | Path | list[str | Path],
    seed: int = 114514,
    verbose: bool = False,
) -> dict[str, Any]:
    if isinstance(targets, (str, Path)):
        targets = [targets]

    result = {"seed": seed}
    torch.manual_seed(seed)

    if verbose:
        tqdm.write(f"seed = {seed}")

    base_model = Model(base)
    result["base"] = base_model.dict()

    if verbose:
        tqdm.write(str(base_model) + "\n")

    map_attn_a = {}
    map_rand_input = {}

    for n in range(3, 11):
        layer = ATTN1.format(n=n, qkv="q")
        hidden_dim, embed_dim = base_model.state_dict[layer].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[n] = evaluate(base_model.state_dict, n, rand_input)
        map_rand_input[n] = rand_input

    del base_model

    for tgt in tqdm(targets, leave=verbose):
        if not check_path(tgt):
            continue

        try:
            target = Model(tgt)
        except Exception:
            tqdm.write(f"Failed to load {tgt}\n")
            continue

        key = Path(target.path).name
        result[key] = target.dict()

        if verbose:
            tqdm.write(str(target))

        sims = {}

        for n in range(3, 11):
            attn_a = map_attn_a[n]
            attn_b = evaluate(target.state_dict, n, map_rand_input[n])
            sim = torch.cosine_similarity(attn_a, attn_b).mean()

            sims[n] = sim.item()

            if verbose:
                tqdm.write(f"{n}: {sims[n]:.2%}")

        result[key]["similarity"] = sims
        mean = np.mean(list(sims.values()))
        result[key]["similarity"]["mean"] = mean

        if verbose:
            tqdm.write(f"mean: {mean:.2%}\n")

    return result
