from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

from src.utils import extract_archive, write_dataset


class IntelImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label


class IntelImgClfDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_data_dir: str = "data/",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.root_data_dir = Path(root_data_dir)
        self.dataset_zip = self.root_data_dir / "intel_imageclf.zip"
        self.dataset_extracted = self.root_data_dir / "intel-image-classification"
        self.dataset_dir = self.root_data_dir / "intel-image-classification" / "dataset"

        self.dataset_extracted.mkdir(parents=True, exist_ok=True)

        # data transformations
        self.train_transforms = A.Compose(
            [
                A.Rotate(limit=5, interpolation=1, border_mode=4),
                A.HorizontalFlip(),
                A.CoarseDropout(2, 8, 8, 1, 8, 8),
                A.RandomBrightnessContrast(brightness_limit=1.5, contrast_limit=0.9),
                A.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )
        self.test_transforms = A.Compose(
            [
                A.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.classes)

    @property
    def classes(self):
        return self.data_train.classes

    @property
    def idx_to_class(self):
        return {k: v for v, k in self.data_train.class_to_idx.items()}

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """

        if not self.dataset_dir.exists():
            # split dataset and save to their directories
            print(f":: Extracting Zip {self.dataset_zip} to {self.dataset_extracted}")
            extract_archive(from_path=self.dataset_zip, to_path=self.dataset_extracted)

            ds = list((self.dataset_extracted / "seg_train" / "seg_train").glob("*/*"))
            ds += list((self.dataset_extracted / "seg_test" / "seg_test").glob("*/*"))
            d_pred = list((self.dataset_extracted / "seg_pred" / "seg_pred").glob("*/"))

            labels = [x.parent.stem for x in ds]
            print(":: Dataset Class Counts: ", Counter(labels))

            d_train, d_test = train_test_split(ds, test_size=0.3, stratify=labels)
            d_test, d_val = train_test_split(
                d_test, test_size=0.5, stratify=[x.parent.stem for x in d_test]
            )

            print(
                "\t:: Train Dataset Class Counts: ",
                Counter(x.parent.stem for x in d_train),
            )
            print(
                "\t:: Test Dataset Class Counts: ",
                Counter(x.parent.stem for x in d_test),
            )
            print(
                "\t:: Val Dataset Class Counts: ", Counter(x.parent.stem for x in d_val)
            )

            print(":: Writing Datasets")
            write_dataset(d_train, self.dataset_dir / "train")
            write_dataset(d_test, self.dataset_dir / "test")
            write_dataset(d_val, self.dataset_dir / "val")
            write_dataset(d_pred, self.dataset_dir / "pred")
        else:
            print(":: Skipping dataset exists")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            trainset = ImageFolder(self.dataset_dir / "train")
            testset = ImageFolder(self.dataset_dir / "test")
            valset = ImageFolder(self.dataset_dir / "val")

            self.data_train, self.data_test, self.data_val = trainset, testset, valset

    def train_dataloader(self):
        return DataLoader(
            dataset=IntelImageDataset(self.data_train, self.train_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=IntelImageDataset(self.data_val, self.test_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=IntelImageDataset(self.data_test, self.test_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    datamodule = IntelImgClfDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.idx_to_class)

    for batch in datamodule.train_dataloader():
        x, y = batch
        print(x.shape, y.shape)
        break
