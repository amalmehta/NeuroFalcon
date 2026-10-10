"""Day 2: GRU decoder.

    python -m neurofalcon.gru     # pretrain 3 seeds, save to models/, score like the baseline

    z_t = (s_t - mean) / std                    z-scored counts, 96 channels
    u_t = A z_t + a                             input layer, 96 -> 32
    h_t = GRU(u_t, h_{t-1})                     64 hidden units
    v_t = (B h_t + b) * vel_std + vel_mean      readout, 64 -> 2

Size and training length were picked on a within-run 80/20 split of held-in
data (validation R², seed 0):
    64 hidden, 600 steps   0.555   (75 s)    <- chosen
    128 hidden, 600 steps  0.572   (204 s)
    64 hidden, 1500 steps  0.492   (overfits)
    128 hidden, 1500 steps 0.524
The Wiener filter scores 0.30 on the same split.

The model is split into named parts (`norm`, `inp`, `gru`, `out`) so
recalibration can choose which of them to update on a new day.

Training samples random windows of `window` bins from the runs. Each window
starts from a zero hidden state, so the first `burn_in` bins are left out of
the loss. At test time the harness streams one bin at a time from the start
of a run and the hidden state is never reset inside a run.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from falcon_challenge.interface import BCIDecoder

from neurofalcon import sessions
from neurofalcon.sessions import CONFIG, Run

torch.set_num_threads(4)


class GRUNet(nn.Module):
    def __init__(self, n_in: int = 96, n_proj: int = 32, n_hidden: int = 64, n_out: int = 2):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))
        self.inp = nn.Linear(n_in, n_proj)
        self.gru = nn.GRU(n_proj, n_hidden, batch_first=True)
        self.out = nn.Linear(n_hidden, n_out)

    def set_input_stats(self, spikes: np.ndarray) -> None:
        spikes = spikes.astype(np.float64)
        std = spikes.std(0)
        std[std == 0] = 1.0
        self.x_mean.copy_(torch.as_tensor(spikes.mean(0)))
        self.x_std.copy_(torch.as_tensor(std))

    def set_output_stats(self, vel: np.ndarray) -> None:
        self.y_mean.copy_(torch.as_tensor(vel.mean(0)))
        self.y_std.copy_(torch.as_tensor(vel.std(0)))

    def forward(self, spikes: torch.Tensor, h: torch.Tensor | None = None):
        """spikes (B, T, 96) counts -> velocity in standard units (B, T, 2), final hidden."""
        z = (spikes - self.x_mean) / self.x_std
        y, h = self.gru(self.inp(z), h)
        return self.out(y), h

    def to_velocity(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean

    @torch.no_grad()
    def predict(self, spikes: np.ndarray) -> np.ndarray:
        """Whole-run prediction from a zero state, (T, 96) -> (T, 2)."""
        self.eval()
        y, _ = self(torch.as_tensor(spikes, dtype=torch.float32)[None])
        return self.to_velocity(y)[0].numpy()


def _windows(runs: Sequence[Run], window: int, batch: int, rng: np.random.Generator):
    """Random (batch, window) slices; a run shorter than `window` is used whole."""
    window = min(window, min(len(r.spikes) for r in runs))
    lengths = np.array([len(r.spikes) - window + 1 for r in runs])
    pick = rng.choice(len(runs), size=batch, p=lengths / lengths.sum())
    xs, ys, ms = [], [], []
    for i in pick:
        r = runs[i]
        t = rng.integers(0, len(r.spikes) - window + 1)
        xs.append(r.spikes[t:t + window])
        ys.append(r.velocity[t:t + window])
        ms.append(r.eval_mask[t:t + window])
    f = lambda a, dt: torch.as_tensor(np.stack(a), dtype=dt)
    return f(xs, torch.float32), f(ys, torch.float32), f(ms, torch.bool), window


def train(
    net: GRUNet,
    runs: Sequence[Run],
    params: Iterable[nn.Parameter] | None = None,
    steps: int = 600,
    lr: float = 3e-3,
    window: int = 100,
    burn_in: int = 25,
    batch: int = 32,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> GRUNet:
    """Minimise MSE in standardised velocity units on eval_mask bins after burn-in."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    params = list(net.parameters() if params is None else params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    net.train()
    for _ in range(steps):
        x, v, m, w = _windows(runs, window, batch, rng)
        y, _ = net(x)
        target = (v - net.y_mean) / net.y_std
        keep = m.clone()
        keep[:, : min(burn_in, w // 4)] = False  # short adaptation windows keep most bins
        loss = ((y - target) ** 2).mean(-1)[keep].mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    net.eval()
    return net


def fit_gru(runs: Sequence[Run], seed: int = 0, **kw) -> GRUNet:
    """Pretrain on held-in runs. Input and output statistics come from the same runs."""
    torch.manual_seed(seed)
    net = GRUNet()
    net.set_input_stats(np.concatenate([r.spikes for r in runs]))
    net.set_output_stats(np.concatenate([r.velocity[r.eval_mask] for r in runs]))
    return train(net, runs, seed=seed, **kw)


def clone(net: GRUNet) -> GRUNet:
    return copy.deepcopy(net)


class GRUDecoder(BCIDecoder):
    """Streams a GRUNet through the harness.

    `nets` is one network for every file, or a dict from recording date
    (e.g. "20201030") to the network recalibrated for that day; dates not in
    the dict fall back to `nets["default"]`.
    """

    def __init__(self, nets: GRUNet | dict[str, GRUNet], batch_size: int = 1):
        super().__init__(task_config=CONFIG, batch_size=batch_size)
        self.nets = nets if isinstance(nets, dict) else {"default": nets}
        self.reset()

    def reset(self, dataset_tags=None):
        tags = dataset_tags or [None] * self.batch_size
        self.active = [self._net_for(t) for t in tags]
        self.h = [None] * len(self.active)

    def _net_for(self, tag) -> GRUNet:
        if tag is None:
            return self.nets["default"]
        date = CONFIG.hash_dataset(tag).split("_")[1]
        return self.nets.get(date, self.nets["default"])

    @torch.no_grad()
    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(neural_observations, dtype=torch.float32)
        out = []
        for i in range(x.shape[0]):
            net = self.active[min(i, len(self.active) - 1)]
            y, self.h[i] = net(x[i][None, None], self.h[i])
            out.append(net.to_velocity(y)[0, 0])
        return torch.stack(out).numpy()

    def on_done(self, dones):
        pass  # M2 is scored continuously; trial ends don't reset state


def save(nets: dict[str, GRUNet], path) -> None:
    torch.save({k: v.state_dict() for k, v in nets.items()}, path)


def load(path) -> dict[str, GRUNet]:
    states = torch.load(path, map_location="cpu")
    nets = {}
    for k, state in states.items():
        nets[k] = GRUNet()
        nets[k].load_state_dict(state)
        nets[k].eval()
    return nets


MODELS = Path(__file__).resolve().parent.parent / "models"


def model_path(seed: int):
    return MODELS / f"GRU held-in seed {seed}.pt"


def pretrained(seeds: int = 3, retrain: bool = False) -> list[GRUNet]:
    """Held-in GRUs, one per seed: loaded from models/ if saved, else trained and saved."""
    held_in = None
    nets = []
    for s in range(seeds):
        path = model_path(s)
        if path.exists() and not retrain:
            nets.append(load(path)["default"])
            continue
        held_in = held_in or sessions.load_split("held-in-calib")
        print(f"pretraining seed {s} on {len(held_in)} held-in runs", flush=True)
        nets.append(fit_gru(held_in, seed=s))
        MODELS.mkdir(exist_ok=True)
        save({"default": nets[-1]}, path)
    return nets


def main(argv=None) -> dict:
    import argparse
    import json

    from neurofalcon import scoring

    p = argparse.ArgumentParser(description="Pretrain the GRU on held-in days and score it zero-shot.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args(argv)
    nets = pretrained(args.seeds, args.retrain)
    per_seed = [{"minival": scoring.minival(GRUDecoder(n)),
                 "zero_shot_held_out": scoring.score(GRUDecoder(n), sessions.files("held-out-calib"))}
                for n in nets]
    mean = lambda f: float(np.mean([f(r) for r in per_seed]))
    std = lambda f: float(np.std([f(r) for r in per_seed]))
    result = {
        "decoder": {"name": "GRU", "proj": 32, "hidden": 64, "steps": 600, "seeds": len(nets)},
        "minival_held_in_r2": {"mean": mean(lambda r: r["minival"]["Held In R2 Mean"]),
                               "std_over_seeds": std(lambda r: r["minival"]["Held In R2 Mean"])},
        "zero_shot_held_out_r2": {"mean": mean(lambda r: r["zero_shot_held_out"]["r2_mean"]),
                                  "std_over_seeds": std(lambda r: r["zero_shot_held_out"]["r2_mean"])},
        "per_seed": per_seed,
    }
    scoring.RESULTS.mkdir(exist_ok=True)
    (scoring.RESULTS / "gru.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))
    return result


if __name__ == "__main__":
    main()
