import json
import logging
from collections import defaultdict

import torch
from tqdm import tqdm

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt
from .utils import setup_logging, to_device, trainer
from .vall_e import get_model

_logger = logging.getLogger(__name__)


def load_engines():
    model = get_model(cfg.model)

    engines = dict(
        model=trainer.Engine(
            model=model,
            config=cfg.ds_cfg,
        ),
    )

    return trainer.load_engines(engines, cfg)


def main():
    setup_logging(cfg.log_dir)

    train_dl, train200_dl, val_dl, test_dl = create_train_val_dataloader()

    def train_feeder(engines, batch, name):
        model = engines["model"]

        if cfg.model == "ar":
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resp_list=batch["resp"],
            )
        elif cfg.model == "nar":
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resps_list=batch["resps"],
            )

        losses = model.gather_attribute("loss")

        loss = torch.stack([*losses.values()]).sum()

        stats = {}
        stats |= {k: v.item() for k, v in losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl):
        log_dir = cfg.log_dir / str(engines.global_step) / name

        model = engines["model"]
        log_dir = cfg.log_dir / str(engines.global_step) / name
        stats = defaultdict(list)
        for batch in tqdm(dl):
            batch: dict
            batch = to_device(batch, cfg.device)

            if cfg.model == "ar":
                resp_list = model(text_list=batch["text"], proms_list=batch["proms"])
                resps_list = [r.unsqueeze(-1) for r in resp_list]
            elif cfg.model == "nar":
                resps_list = model(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resp_list=batch["resp"],
                )
            else:
                raise NotImplementedError(cfg.model)

            losses = model.gather_attribute("loss")
            batch_stats = {k: v.item() for k, v in losses.items()}
            for k, v in batch_stats.items():
                stats[k].append(v)

            for path, ref, hyp in zip(batch["path"], batch["resps"], resps_list):
                relpath = path.relative_to(cfg.data_root)
                hyp_path = (log_dir / "hyp" / relpath).with_suffix(".wav")
                ref_path = (log_dir / "ref" / relpath).with_suffix(".wav")
                hyp_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                qnt.decode_to_file(ref, ref_path)
                if len(hyp) > 0:
                    qnt.decode_to_file(hyp, hyp_path)

        stats = {k: sum(v) / len(v) for k, v in stats.items()}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        _logger.info(f"Eval: {stats}.")

        _logger.info(f"{json.dumps(stats)}.")

    def eval_fn(engines):
        run_eval(engines, "train200", train200_dl)
        run_eval(engines, "val", val_dl)
        run_eval(engines, "test", test_dl)

    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        train_feeder=train_feeder,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    main()