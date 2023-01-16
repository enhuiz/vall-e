import torch
from torch import Tensor

from .base import Base


class NAR(Base):
    @property
    def n_resp_levels(self):
        return 7

    @property
    def casual(self):
        return False

    @property
    def use_stop_token(self):
        return False

    @property
    def norm_type(self):
        return "adaln"

    @property
    def resp_loss_only(self):
        return True

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        sampling_temperature: float = 0.2,
    ):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' l] * b, l=8
            resps_list: [t'' l] * b, l=1 or 8, 1 for testing and 8 for training.
        Returns:
            [t'' l], l=8 if testing. empty list will be returned during training.
        """

        n_levels_set = {r.shape[-1] for r in resps_list}

        if len(n_levels_set) > 1:
            raise ValueError(f"Please give only one level, got {n_levels_set}.")

        n_levels = next(iter(n_levels_set))

        device = text_list[0].device

        if n_levels == self.n_resp_levels + 1:
            assert resps_list is not None

            quant_levels = torch.randint(0, self.n_resp_levels, (len(resps_list),))

            prev_list = [o[..., : l + 1] for o, l in zip(resps_list, quant_levels)]
            targ_list = [o[..., l + 1] for o, l in zip(resps_list, quant_levels)]

            quant_levels = quant_levels.to(device=device)

            _ = super().forward(
                text_list,
                proms_list,
                prev_list,
                targ_list,
                return_all_resp=True,
                shift_targ_list=False,
                quant_levels=quant_levels,
            )

            # Yes, just nothing as we are training
            prev_list = []
        else:
            prev_list = resps_list

            while True:
                level = prev_list[0].shape[-1] - 1

                if level >= self.n_resp_levels:
                    break

                quant_levels = torch.full((len(text_list),), level, device=device)

                resp_list = super().forward(
                    text_list,
                    proms_list,
                    prev_list,
                    return_all_resp=True,
                    shift_targ_list=False,
                    quant_levels=quant_levels,
                    sampling_temperature=sampling_temperature,
                )

                prev_list = [
                    torch.cat([rs, r.unsqueeze(-1)], dim=-1)
                    for rs, r in zip(prev_list, resp_list)
                ]

        return prev_list


def example_usage():
    from functools import partial
    from pathlib import Path

    from einops import repeat

    from ..emb.qnt import decode_to_file
    from ..utils import gather_attribute

    device = "cuda"

    resps = torch.load("data/test/test.qnt.pt")[0].to(device)
    num_qnts = 1024

    model = NAR(num_qnts).to(device)

    text_list = [
        torch.tensor([2, 3], device=device),
    ]

    x8 = partial(repeat, pattern="t -> t l", l=8)
    proms_list = [
        x8(torch.tensor([2, 3], device=device)),
    ]

    resps_x1_list = [
        resps[:1].t().to(device),
    ]

    resps_x8_list = [
        resps.t().to(device),
    ]

    codes = model(
        text_list,
        proms_list,
        resps_list=resps_x1_list,
        sampling_temperature=0.2,
    )[0]

    decode_to_file(
        codes,
        Path("data/test/test.nar.init.wav"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(200):
        optimizer.zero_grad()

        _ = model(text_list, proms_list, resps_list=resps_x8_list)

        losses = gather_attribute(model, "loss")
        loss = sum(losses.values())
        loss.backward()

        optimizer.step()

        if i % 20 == 0:
            stats = {k: v.item() for k, v in losses.items()}
            stats["loss"] = loss.item()
            print(f"iter={i}, {stats}.")

    for i in range(1, 8):
        resps_list = [
            resps[:i].t().to(device),
        ]

        codes = model(
            text_list,
            proms_list,
            resps_list=resps_list,
            sampling_temperature=0.2,
        )[0]

        decode_to_file(
            codes,
            Path(f"data/test/test.nar.1-{i}.wav"),
        )


if __name__ == "__main__":
    example_usage()
