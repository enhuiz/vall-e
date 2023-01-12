import torch
from einops import rearrange
from torch import Tensor

from .base import Base


class NAR(Base):
    @property
    def n_levels(self):
        return 7

    @property
    def casual(self):
        return False

    @property
    def use_stop_token(self):
        return False

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        *,
        resp_list: list[Tensor] | None = None,
        resps_list: list[Tensor] | None = None,
    ):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' k] * b
            resp_list: [t'] * b, quants at level 0.
            resps_list: [t''] * b, 8 quantization levels for training.
        Returns:
            y: logits of last output, b k
        """
        if (resp_list is None) == (resps_list is None):
            raise ValueError(
                "Given one and only one, either resp_list (generation) or resps_list (training)."
            )

        if resps_list is not None:
            levels = {r.shape[-1] for r in resps_list}
            if any(level != self.n_levels + 1 for level in levels):
                raise ValueError(
                    f"resps_list should have exactly {self.n_levels + 1} levels, but got {levels}."
                )

        if resp_list is not None:
            hyp_resp_lists = [resp_list]
            for i in range(self.n_levels):
                hyp_resp_list = super().forward(
                    text_list,
                    proms_list,
                    hyp_resp_lists[-1],
                    return_all_resp=True,
                    shift_targ_list=False,
                    quant_level=i,
                )
                hyp_resp_lists.append(hyp_resp_list)
        else:
            assert resps_list is not None

            loss = {}
            resp_list = [o[..., 0] for o in resps_list]
            hyp_resp_lists = [resp_list]
            for i in range(self.n_levels):
                resp_list = [o[..., 0] for o in resps_list]
                next_resp_list = [o[..., i + 1] for o in resps_list]
                hyp_resp_list = super().forward(
                    text_list,
                    proms_list,
                    resp_list,
                    next_resp_list,
                    return_all_resp=True,
                    shift_targ_list=False,
                    quant_level=i,
                )
                hyp_resp_lists.append(hyp_resp_list)
                loss |= {f"l{i}": self.loss}
                del self.loss
            self.loss = loss

        hyp_resps_list = [
            *map(lambda ts: torch.stack(ts, dim=-1), zip(*hyp_resp_lists))
        ]

        return hyp_resps_list


def example_usage():
    from functools import partial

    import soundfile
    from einops import repeat

    from ..emb.qnt import decode
    from ..utils import gather_attribute

    device = "cuda"

    resps = torch.load("data/test/test.qnt.pt")[0].to(device)
    num_qnts = 1024

    model = NAR(num_qnts).to(device)

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
        resps[0].to(device),
    ]

    resps_list = [
        x8(torch.tensor([1, 2, 3], device=device)),
        resps.t().to(device),
    ]

    out = model(text_list, proms_list, resp_list=resp_list)
    codes = rearrange(out[1], "t k -> 1 k t")
    print(codes)
    wavs, sr = decode(codes)
    soundfile.write("data/test/test.nar.init.wav", wavs.cpu()[0, 0], sr)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        _ = model(text_list, proms_list, resps_list=resps_list)

        losses = gather_attribute(model, "loss")
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            stats = {k: v.item() for k, v in losses.items()}
            stats["loss"] = loss.item()
            print(f"iter={i}, {stats}.")

    out = model(text_list, proms_list, resp_list=resp_list)
    codes = rearrange(out[1], "t k -> 1 k t")
    wavs, sr = decode(codes)
    soundfile.write("data/test/test.nar.recon.wav", wavs.cpu()[0, 0], sr)


if __name__ == "__main__":
    example_usage()
