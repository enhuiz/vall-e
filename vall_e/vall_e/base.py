import math
from dataclasses import dataclass
from functools import partial
from typing import Literal, overload

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (t b 1)
    m = rearrange(m, pattern)
    return x, m


class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])  # t d
        e = e[None]  # b t d
        x = x + e
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, casual):
        super().__init__()
        assert d_model % num_heads == 0
        dim_head = d_model // num_heads
        self.casual = casual
        self.num_heads = num_heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.num_heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        e = einsum("b i h d, b j h d -> b i j h", q, k)
        e = e * self.scale

        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        if self.casual:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)  # Normalize on j, i.e. key

        o = einsum("b i j h, b j h d -> b i h d", a, v)
        o = o.flatten(-2)
        o = self.to_out(o)  # b t c

        o = o * m

        return o


class PrenormResidual(nn.Module):
    def __init__(self, block, d_model, dropout, requires_mask=False):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, m):
        opts = {"m": m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x) * m, **opts))
        return x * m


class Block(nn.Sequential):
    def __init__(self, d_model, num_heads, dropout, casual):
        super().__init__()
        self.attn = PrenormResidual(
            Attention(d_model, num_heads, casual),
            d_model=d_model,
            dropout=dropout,
            requires_mask=True,
        )
        self.ffn = PrenormResidual(
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            ),
            d_model=d_model,
            dropout=dropout,
        )

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t 1)
        """
        x = self.attn(x, m)
        x = self.ffn(x, m)
        return x


class Embedding(nn.Embedding):
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        if len(x) == 0:
            return []
        return super().forward(torch.cat(x)).split([*map(len, x)])


def _join(x: tuple[Tensor], sep: Tensor):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret


class Base(nn.Module):
    @property
    def casual(self) -> bool:
        raise NotImplementedError

    @property
    def n_levels(self) -> int:
        raise NotImplementedError

    @property
    def use_stop_token(self) -> bool:
        raise NotImplementedError

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        n_levels = self.n_levels
        casual = self.casual

        n_stop_tokens = 1 if self.use_stop_token else 0
        n_resp_tokens = n_tokens + n_stop_tokens

        self.text_emb = Embedding(n_tokens, d_model)
        self.prom_emb = Embedding(n_tokens, d_model)

        # +1 to include the stop token
        self.resp_embs = nn.ModuleList(
            [Embedding(n_resp_tokens, d_model) for _ in range(n_levels)]
        )

        self.sin_emb = SinusodialEmbedding(d_model)

        self.sep = nn.Parameter(torch.randn(d_model))

        blocks = [Block(d_model, n_heads, p_dropout, casual) for _ in range(n_layers)]
        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Linear(d_model, n_resp_tokens)

    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens

    @property
    def ignore_index(self):
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: Tensor | None):
        if sep is None:
            cat = torch.cat
        else:
            cat = partial(_join, sep=sep)
        return [*map(cat, zip(*l))]

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        prom_list: list[Tensor],
        resp_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_level: int = 0,
        shift_targ_list: bool = False,
        return_all_resp: Literal[False] = False,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        prom_list: list[Tensor],
        resp_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_level: int = 0,
        shift_targ_list: bool = False,
        return_all_resp: Literal[True] = True,
    ) -> list[Tensor]:
        ...

    def forward(
        self,
        text_list: list[Tensor],
        prom_list: list[Tensor],
        resp_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_level: int = 0,
        shift_targ_list: bool = False,
        return_all_resp: bool = False,
    ):
        """
        Args:
            text_list: [t] * b
            prom_list: [t'] * b
            resp_list: [t''] * b, one quantization level only
            targ_list: [t''] * b, one quantization level only, when given, loss will be computed
            quant_level: specify which quant_level to feed forward, used in NAR mode.
            shift_targ_list: whether to shift target list when computing loss. True if AR.
            return_all_resp: True if NAR.
        Returns:
            y: sampled tokens
        """
        x_list = self._samplewise_merge_tensors(
            self.text_emb(text_list),
            self.prom_emb(prom_list),
            self.resp_embs[quant_level](resp_list),
            sep=self.sep,
        )

        x, m = list_to_tensor(x_list)
        x = self.sin_emb.add_pe(x)

        for block in self.blocks:
            x = block(x, m)

        h = self.classifier(x) * m

        # Remove padding
        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]

        if targ_list is not None:
            if any([l == 0 for l in map(len, targ_list)]):
                raise ValueError("Cannot compute loss given empty targ_list.")

            device = h.device

            ignore_sep = torch.tensor(self.ignore_index, device=device)
            text_prom_list = self._samplewise_merge_tensors(
                text_list, prom_list, sep=ignore_sep
            )

            # Make every token earlier as it is future that is unknown
            for i in range(len(text_prom_list)):
                text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                text_prom_list[i][-1] = self.ignore_index

            if shift_targ_list:
                # Also make target earlier if in autoregressive mode
                targ_list = [*targ_list]
                for i in range(len(targ_list)):
                    targ_list[i] = targ_list[i].roll(-1, dims=0)
                    targ_list[i][-1] = self.stop_token

            y_list = self._samplewise_merge_tensors(
                text_prom_list, targ_list, sep=ignore_sep
            )

            self.loss = dict(
                nll=F.cross_entropy(
                    torch.cat(h_list),
                    torch.cat(y_list),
                    ignore_index=self.ignore_index,
                )
            )

        if return_all_resp:
            logits = [hi[-li:] for hi, li in zip(h_list, map(len, resp_list))]
            ret = [Categorical(logits=hi).sample() for hi in logits]
        else:
            logits = torch.stack([hi[-1] for hi in h_list])
            ret = Categorical(logits=logits).sample()

        return ret
