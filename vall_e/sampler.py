"""
A sampler that balances data by key_fns.

MIT License

Copyright (c) 2023 Zhe Niu

niuzhe.nz@outlook.com
"""

import random


class Sampler:
    def __init__(self, l, key_fns):
        self.tree = self._build(l, key_fns)

    def _build(self, l, key_fns) -> dict[dict, list]:
        if not key_fns:
            return l

        tree = {}

        key_fn, *key_fns = key_fns

        for x in l:
            k = key_fn(x)

            if k in tree:
                tree[k].append(x)
            else:
                tree[k] = [x]

        for k in tree:
            tree[k] = self._build(tree[k], key_fns)

        return tree

    def _sample(self, tree: dict | list):
        if isinstance(tree, list):
            ret = random.choice(tree)
        else:
            key = random.choice([*tree.keys()])
            ret = self._sample(tree[key])
        return ret

    def sample(self):
        return self._sample(self.tree)
