"""
PyTorch DataLoader utilities.

Responsibilities
----------------
- Create reproducible DataLoaders.
- Read all settings from config.yaml.
- Support train/validation/test loaders.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from utils.seed import create_generator, seed_worker


class DataLoaderFactory:
    """
    Factory class for creating project DataLoaders.
    """

    def __init__(self, cfg):

        self.cfg = cfg

        self.generator = create_generator(
            cfg.project.seed
        )

    def _persistent_workers(self):

        if self.cfg.project.num_workers == 0:
            return False

        value = self.cfg.dataloader.persistent_workers

        if isinstance(value, str):
            return value.lower() == "true"

        return bool(value)

    def create(
        self,
        dataset,
        split: str,
    ):
        """
        Parameters
        ----------
        dataset
            PyTorch Dataset

        split
            train | validation | test
        """

        split = split.lower()

        if split == "train":
            shuffle = self.cfg.dataloader.shuffle_train
            drop_last = self.cfg.dataloader.drop_last_train

        elif split == "validation":
            shuffle = self.cfg.dataloader.shuffle_validation
            drop_last = False

        elif split == "test":
            shuffle = self.cfg.dataloader.shuffle_test
            drop_last = False

        else:
            raise ValueError(
                f"Unknown split: {split}"
            )

        return DataLoader(
            dataset=dataset,

            batch_size=self.cfg.dataloader.batch_size,

            shuffle=shuffle,

            drop_last=drop_last,

            num_workers=self.cfg.project.num_workers,

            pin_memory=self.cfg.project.pin_memory,

            persistent_workers=self._persistent_workers(),

            worker_init_fn=seed_worker,

            generator=self.generator,

            prefetch_factor=(
                self.cfg.dataloader.prefetch_factor
                if self.cfg.project.num_workers > 0
                else None
            ),
        )

    def train(self, dataset):
        return self.create(dataset, "train")

    def validation(self, dataset):
        return self.create(dataset, "validation")

    def test(self, dataset):
        return self.create(dataset, "test")