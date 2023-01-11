import argparse
from functools import cache
from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor
from tqdm import tqdm


@cache
def _load_model(device="cuda"):
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    return model


@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
    """
    Args:
        codes: (b k t)
    """
    assert codes.dim() == 3
    model = _load_model(device)
    return model.decode([(codes, None)]), model.sample_rate


def replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


@torch.inference_mode()
def encode(wav, sr, device="cuda"):
    """
    Args:
        wav: (t)
        sr: int
    """
    model = _load_model(device)
    wav = wav.unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    encoded_frames = model.encode(wav)
    qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b k t)
    return qnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--suffix", default=".wav")
    args = parser.parse_args()

    paths = [*args.folder.rglob(f"*{args.suffix}")]

    for path in tqdm(paths):
        out_path = replace_file_extension(path, ".qnt.pt")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] == 2:
            wav = wav[:1]
        qnt = encode(wav, sr)
        torch.save(qnt.cpu(), out_path)


if __name__ == "__main__":
    main()
