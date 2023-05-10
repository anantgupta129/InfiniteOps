import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from git.repo.base import Repo
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule


dvc_repo_url = os.environ.get("DVC_REPO_URL")
dvc_branch = os.environ.get("DVC_BRANCH")

git_user = os.environ.get("GIT_USER", "sagemaker")
git_email = os.environ.get("GIT_EMAIL", "sagemaker-processing@example.com")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "intel_imageclf.zip"
annotations_path = ml_root / "input" / "annotations"
git_path = ml_root / "kaggle_intel_image_classification"


def configure_git():
    subprocess.check_call(["git", "config", "--global", "user.email", f'"{git_email}"'])
    subprocess.check_call(["git", "config", "--global", "user.name", f'"{git_user}"'])


def clone_dvc_git_repo():
    print(f"\t:: Cloning repo: {dvc_repo_url}")

    repo = Repo.clone_from(dvc_repo_url, git_path.absolute())

    return repo


def sync_data_with_dvc(repo):
    os.chdir(git_path)
    print(f":: Create branch {dvc_branch}")
    try:
        repo.git.checkout("-b", dvc_branch)
        print(f"\t:: Create a new branch: {dvc_branch}")
    except:
        repo.git.checkout(dvc_branch)
        print(f"\t:: Checkout existing branch: {dvc_branch}")
    print(":: Add files to DVC")

    subprocess.check_call(["dvc", "add", "dataset"])

    repo.git.add(all=True)
    repo.git.commit("-m", f"'add data for {dvc_branch}'")

    print("\t:: Push data to DVC")
    subprocess.check_call(["dvc", "push"])

    print("\t:: Push dvc metadata to git")
    repo.remote(name="origin")
    repo.git.push("--set-upstream", repo.remote().name, dvc_branch, "--force")

    sha = repo.head.commit.hexsha

    print(f":: Commit Hash: {sha}")


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data(dataset_zip=dataset_zip, storage_dir=git_path, annotations_path=annotations_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup git
    print(":: Configuring Git")
    configure_git()
    print(":: Cloning Git")
    repo = clone_dvc_git_repo()

    main()

    print(":: copy data to train")
    subprocess.check_call(
        "cp -r /opt/ml/processing/kaggle_intel_image_classification/dataset/* /opt/ml/processing/dataset/",
        shell=True,
    )

    print(":: Sync Processed Data to Git & DVC")
    sync_data_with_dvc(repo)
    print(":: finished pre processing dataset")
