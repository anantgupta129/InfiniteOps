import gc
import json
import os
import tarfile
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.models.timm_module import LitModule

log = utils.get_pylogger(__name__)

ml_root = Path("/opt/ml")

model_artifacts = ml_root / "processing" / "model"
dataset_dir = ml_root / "processing" / "test"


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")

    cfg.datamodule.train_data_dir = dataset_dir.absolute()
    cfg.datamodule.test_data_dir = dataset_dir.absolute()
    cfg.datamodule.val_data_dir = dataset_dir.absolute()
    cfg.datamodule.num_workers = os.cpu_count()

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    model: LightningModule = LitModule.load_from_checkpoint(checkpoint_path="last.ckpt")

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
    }

    log.info("Starting testing!")
    metric_dict = utils.calc_metric(model, datamodule)

    report_dict = {
        "multiclass_classification_metrics": {**metric_dict},
    }

    gc.collect()

    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)

    out_path = eval_folder / "evaluation.json"

    print(f":: Writing to {out_path.absolute()}")

    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
