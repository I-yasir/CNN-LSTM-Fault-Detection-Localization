
"""
Verify the complete data pipeline.

This script validates:
1. Configuration
2. Dataset
3. Dataset split
4. Normalization
5. Lazy dataset
6. DataLoader

Run:

    python src/verify_data.py
"""

from pathlib import Path

import torch

from dataset.dataloader import DataLoaderFactory
from dataset.lazy_dataset import LazyFaultDataset
from dataset.normalization import ZScoreNormalizer
from dataset.splitter import DatasetSplitter
from dataset.validator import DatasetValidator
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def print_batch_statistics(signals: torch.Tensor):

    print("\nBatch Statistics")
    print("-" * 40)

    print(f"Shape        : {tuple(signals.shape)}")
    print(f"Mean         : {signals.mean():.6f}")
    print(f"Std          : {signals.std():.6f}")
    print(f"Min          : {signals.min():.6f}")
    print(f"Max          : {signals.max():.6f}")


def main():

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    cfg = load_config("configuration/config.yaml")

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    set_seed(
        seed=cfg.project.seed,
        deterministic=cfg.project.deterministic,
        benchmark=cfg.project.benchmark,
    )

    # ------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------
    logger = get_logger(
        cfg.logging.directory,
        cfg.logging.level,
    )

    # ------------------------------------------------------------
    # Validate Dataset
    # ------------------------------------------------------------
    validator = DatasetValidator(cfg, logger)

    validator.validate()

    # ------------------------------------------------------------
    # Dataset Split
    # ------------------------------------------------------------
    splitter = DatasetSplitter(cfg, logger)

    split = splitter.create()

    splitter.summary()

    # ------------------------------------------------------------
    # Training Dataset (raw)
    # ------------------------------------------------------------
    train_dataset = LazyFaultDataset(
        cfg,
        split["train"],
        transform=None,
    )

    # ------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------
    norm_path = Path(cfg.normalization.save_path)

    if norm_path.exists():

        logger.info("Loading normalization parameters...")

        normalizer = ZScoreNormalizer.load(norm_path)

    else:

        logger.info("Computing normalization parameters...")

        normalizer = ZScoreNormalizer()

        normalizer.fit(train_dataset)

        normalizer.save(norm_path)

        logger.info("Normalization parameters saved.")

    logger.info("Mean : %s", normalizer.mean)

    logger.info("Std  : %s", normalizer.std)

    # ------------------------------------------------------------
    # Normalized datasets
    # ------------------------------------------------------------
    train_dataset = LazyFaultDataset(
        cfg,
        split["train"],
        transform=normalizer,
    )

    val_dataset = LazyFaultDataset(
        cfg,
        split["validation"],
        transform=normalizer,
    )

    test_dataset = LazyFaultDataset(
        cfg,
        split["test"],
        transform=normalizer,
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    factory = DataLoaderFactory(cfg)

    train_loader = factory.train(train_dataset)

    val_loader = factory.validation(val_dataset)

    test_loader = factory.test(test_dataset)

    # ------------------------------------------------------------
    # Read one batch
    # ------------------------------------------------------------
    train_signals, train_class, train_location = next(iter(train_loader))

    val_signals, val_class, val_location = next(iter(val_loader))

    test_signals, test_class, test_location = next(iter(test_loader))

    # ------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)

    print("\nTrain Batch")
    print(f"Signals   : {train_signals.shape}")
    print(f"Classes   : {train_class.shape}")
    print(f"Locations : {train_location.shape}")

    print("\nValidation Batch")
    print(f"Signals   : {val_signals.shape}")

    print("\nTest Batch")
    print(f"Signals   : {test_signals.shape}")

    print_batch_statistics(train_signals)

    print("\nExample Labels")
    print("-" * 40)
    print("Fault Class    :", train_class[:10].tolist())
    print("Fault Location :", train_location[:10].tolist())

    print("\n" + "=" * 60)
    print("DATA PIPELINE VERIFIED SUCCESSFULLY")
    print("=" * 60)

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()


if __name__ == "__main__":
    main()
