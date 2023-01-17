import argparse

import torch

from .data import VALLEDatset, create_train_val_dataloader
from .train import load_engines


def main():
    parser = argparse.ArgumentParser("Save trained model to path.")
    parser.add_argument("path")
    args = parser.parse_args()

    engine = load_engines()
    model = engine["model"].module.cpu()
    train_dl, *_ = create_train_val_dataloader()
    assert isinstance(train_dl.dataset, VALLEDatset)
    model.phone_symmap = train_dl.dataset.phone_symmap
    model.spkr_symmap = train_dl.dataset.spkr_symmap
    torch.save(model, args.path)
    print(args.path, "saved.")


if __name__ == "__main__":
    main()
