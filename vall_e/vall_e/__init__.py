from ..config import cfg
from .ar import AR
from .nar import NAR


def get_model(name):
    name = name.lower()

    if name.startswith("ar"):
        Model = AR
    elif name.startswith("nar"):
        Model = NAR
    else:
        raise ValueError("Model name should start with AR or NAR.")

    if "-quarter" in name:
        model = Model(
            cfg.num_tokens,
            d_model=256,
            n_heads=4,
            n_layers=12,
        )
    elif "-half" in name:
        model = Model(
            cfg.num_tokens,
            d_model=512,
            n_heads=8,
            n_layers=12,
        )
    else:
        if name not in ["ar", "nar"]:
            raise NotImplementedError(name)

        model = Model(
            cfg.num_tokens,
            d_model=1024,
            n_heads=16,
            n_layers=12,
        )

    return model
