# VALL-E

An unofficial (toy) implementation of [VALL-E](https://valle-demo.github.io/), based on the [encodec](https://github.com/facebookresearch/encodec) tokenizer.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/enhuiz)

## Install

```
pip install git+https://github.com/enhuiz/vall-e
```

## Data Preparation

1. Put your data into a folder, e.g. `data/your_data`. Audio files should be named with the suffix `.wav` and text files with `.normalized.txt`.

2. Quantize the data:

```
python -m vall_e.emb.qnt data/your_data
```

3. Generate phonemes based on the text:

```
python -m vall_e.emb.g2p data/your_data
```

4. Customize your configuration by creating `config/your_data/ar.yml` and `config/your_data/nar.yml`. Refer to the example configs in `config/test` and `vall_e/config.py` for details.

5. Train the AR or NAR model using the following scripts:

```
python -m vall_e.train yaml=config/your_data/ar_or_nar.yml
```

## TODO

- [x] AR model for the first quantizer
- [x] Audio decoding from tokens
- [x] NAR model for the rest quantizers
- [x] Trainers for both models
- [ ] Pre-trained checkpoint and demos on LibriTTS
