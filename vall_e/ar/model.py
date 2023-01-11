import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def _list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
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


class CasualAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        dim_head = d_model // num_heads
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
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = PrenormResidual(
            CasualAttention(d_model, num_heads),
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


class EmbeddingWithPE(nn.Module):
    def __init__(self, num_tokens, token_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, token_dim)
        self.sin_emb = SinusodialEmbedding(token_dim)

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        if len(x_list) == 0:
            return []

        x = pad_sequence(x_list, batch_first=True)  # b t
        x = self.embedding(x)  # b t d
        x = self.sin_emb.add_pe(x)
        x_list = [xi[:li] for xi, li in zip(x, map(len, x_list))]

        return x_list


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


class VALLEAR(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        num_layers=12,
    ):
        super().__init__()
        # Here, simply use num_tokens := max(num_text_tokens, num_prompt_tokens, num_output_tokens)
        self.text_emb = EmbeddingWithPE(num_tokens, d_model)
        self.prompt_emb = EmbeddingWithPE(num_tokens, d_model)
        self.output_emb = EmbeddingWithPE(num_tokens, d_model)
        self.sep = nn.Parameter(torch.randn(d_model))  # start of sequence token
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_tokens)

    @property
    def _ignore_index(self):
        return -100

    @staticmethod
    def _elementwise_merge_tensors(*l, sep):
        return [*map(lambda ts: _join(ts, sep), zip(*l))]

    def forward(
        self,
        text_list: list[Tensor],
        prompt_list: list[Tensor],
        output_list: list[Tensor],
        compute_loss: bool = True,
    ) -> Tensor:
        """
        Args:
            text_list: b t d
            prompt_list: b t d
            output_list: b t d
        Returns:
            y: logits of last output, b k
        """
        device = text_list[0].device

        x_list = self._elementwise_merge_tensors(
            self.text_emb(text_list),
            self.prompt_emb(prompt_list),
            self.output_emb(output_list),
            sep=self.sep,
        )

        x, m = _list_to_tensor(x_list)

        for block in self.blocks:
            x = block(x, m)

        h = self.fc(x) * m

        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]

        if compute_loss and len(output_list) > 0:
            y_list = self._elementwise_merge_tensors(
                text_list,
                prompt_list,
                output_list,
                sep=torch.tensor(self._ignore_index, device=device),
            )

            # make y_list earlier as it is future that is unknown
            for i in range(len(y_list)):
                y_list[i] = y_list[i].roll(-1, dims=0)
                y_list[i][-1] = self._ignore_index

            self.loss = dict(
                nll=F.cross_entropy(
                    torch.cat(h_list),
                    torch.cat(y_list),
                    ignore_index=self._ignore_index,
                )
            )

        logits = torch.stack([hi[-1] for hi in h_list])

        return logits

    @staticmethod
    def _prune(l: list[int], stop_token: int | None):
        if stop_token is None:
            return l
        n = next((i for i, x in enumerate(l) if x == stop_token), None)
        if n is not None:
            l = l[:n]
        return l

    def generate(
        self,
        text_list: list[Tensor],
        prompt_list: list[Tensor],
        max_steps: int = 1000,
        stop_token: int | None = None,
    ):
        device = text_list[0].device
        output_list: list[Tensor] = [
            torch.zeros(0, device=device).long() for _ in text_list
        ]
        stopped = [False] * len(text_list)
        for _ in trange(max_steps):
            logits = self.forward(
                text_list,
                prompt_list,
                output_list,
                compute_loss=False,
            )
            o = Categorical(logits=logits).sample()
            for i, oi in enumerate(o):
                if oi.item() == stop_token:
                    stopped[i] = True
                output_list[i] = torch.cat([output_list[i], oi[None]])
            if all(stopped):
                break
        pruned = [self._prune(o.tolist(), stop_token) for o in output_list]
        return pruned


def example_usage():
    device = "cuda"

    test_qnt = torch.load("data/test/test.qnt.pt")[0, 0].to(device)
    num_qnts = 1024 + 1
    eoq = num_qnts - 1

    model = VALLEAR(num_qnts).to(device)

    text_list = [
        torch.tensor([1, 2, 3], device=device),
        torch.tensor([2, 3], device=device),
    ]

    prompt_list = [
        torch.tensor([1, 2, 3], device=device),
        torch.tensor([2, 3], device=device),
    ]

    output_list = [
        torch.tensor([1, 2, 3, eoq], device=device),
        torch.tensor([*test_qnt, eoq], device=device),
    ]

    out = model.generate(
        text_list,
        prompt_list,
        max_steps=200,
        stop_token=eoq,
    )

    print(test_qnt)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        _ = model(text_list, prompt_list, output_list)

        losses = model.loss
        sum(losses.values()).backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"iter={i}, {losses}.")

    out = model.generate(
        text_list,
        prompt_list,
        max_steps=200,
        stop_token=eoq,
    )

    print(out)


if __name__ == "__main__":
    example_usage()
