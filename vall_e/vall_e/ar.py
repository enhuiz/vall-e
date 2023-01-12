import torch
from einops import rearrange
from torch import Tensor
from tqdm import trange

from .base import Base


class AR(Base):
    @property
    def n_levels(self):
        return 1

    @property
    def casual(self):
        return True

    @property
    def use_stop_token(self):
        return True

    def _prune(self, l: Tensor):
        indices = (l == self.stop_token).nonzero()
        if len(indices) == 0:
            return l
        return l[: indices.min().item()]

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resp_list: list[Tensor] | None = None,
        max_steps: int = 1000,
    ):
        if resp_list is not None:
            return super().forward(
                text_list,
                proms_list,
                resp_list,
                resp_list,
                quant_level=0,
                shift_targ_list=True,
                return_all_resp=False,
            )
        else:
            return self._generate(text_list, proms_list, max_steps)

    def _generate(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        max_steps: int,
    ):
        device = text_list[0].device
        resp_list: list[Tensor] = [
            torch.zeros(0, device=device).long() for _ in text_list
        ]
        stopped = [False] * len(text_list)
        for _ in trange(max_steps):
            r = super().forward(text_list, proms_list, resp_list)
            for i, ri in enumerate(r):
                if ri.item() == self.stop_token:
                    stopped[i] = True
                resp_list[i] = torch.cat([resp_list[i], ri[None]])
            if all(stopped):
                break
        pruned = [self._prune(r) for r in resp_list]
        return pruned


def example_usage():
    from functools import partial

    import soundfile
    from einops import repeat

    device = "cuda"

    qnt = torch.load("data/test/test.qnt.pt")[0, 0].to(device)
    num_qnts = 1024

    model = AR(num_qnts).to(device)

    text_list = [
        torch.tensor([1, 2, 3], device=device),
        torch.tensor([2, 3], device=device),
    ]

    x8 = partial(repeat, pattern="t -> t q", q=8)
    proms_list = [
        x8(torch.tensor([1, 2, 3], device=device)),
        x8(torch.tensor([2, 3], device=device)),
    ]

    resp_list = [
        torch.tensor([1, 2, 3], device=device),
        qnt.to(device),
    ]

    out = model(text_list, proms_list, max_steps=200)

    print(out)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        _ = model(text_list, proms_list, resp_list)

        losses = model.loss
        sum(losses.values()).backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"iter={i}, {losses}.")

    out = model(text_list, proms_list, max_steps=200)

    print(qnt)
    print(out)

    from ..emb.qnt import decode

    codes = rearrange(out[1], "t -> 1 1 t")
    wavs, sr = decode(codes)
    soundfile.write("data/test/test.ar.recon.wav", wavs.cpu()[0, 0], sr)


if __name__ == "__main__":
    example_usage()
