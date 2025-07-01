"""Manual test to check the speed of the dataset loading process.

"""

import sys
import time

import torch
from tqdm import tqdm
import data


def test_dataset_speed():
    # Arrange
    BATCH_SIZE = 512
    dataset = data.get_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )

    # Act
    start_time = time.perf_counter()
    for _ in tqdm(dataloader):
        pass
    elapsed = time.perf_counter() - start_time

    # Announce
    print(
        f"Time taken to load {len(dataloader) * BATCH_SIZE} samples: {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    # Run the test
    test_dataset_speed()
    sys.exit(0)
