"""Manages data on disk."""

import logging
import pickle
import random
from collections import defaultdict
import sys

import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PATH = "~/gcs/sandbox-michael-menzel-data-us-central1/FaceID-550+vggface"
_QUICK_PATH = (
    "~/gcs/sandbox-michael-menzel-data-us-central1/FaceID-550+vggface-mini"
)


def get_dataset(quick: bool = False):
    """Load dataset from path with transforms."""

    path = _QUICK_PATH if quick else _PATH

    def pad_to_square(img):
        w, h = img.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return torchvision.transforms.functional.pad(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=0
        )

    transforms = v2.Compose(
        [
            v2.Lambda(pad_to_square),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            v2.Resize((224, 224)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(
        path,
        transform=transforms,
        target_transform=torch.tensor,
    )

    return dataset


def get_splits(quick: bool, seed=42):
    """Returns a deterministic split of the dataset into train and test sets.

    Each class will be represented in both sets, and the split will be
    90% train and 10% test.
    """
    random.seed(seed)
    dataset = get_dataset(quick=quick)

    # Check cache
    path = f"/tmp/labels{'-quick' if quick else ''}.pkl"
    try:
        with open(path, "rb") as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        # Cache miss
        logger.info("Cache miss, loading labels from dataset")
        labels = []
        for idx, (_, label) in enumerate(tqdm(dataset, disable=not sys.stdout.isatty())):
            labels.append(label.item())
        pickle.dump(labels, open(path, "wb"))
        with open(path, "rb") as f:
            labels = pickle.load(f)

    label_to_indexes = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indexes[label].append(idx)

    train_indexes, test_indexes = [], []
    for label, indexes in label_to_indexes.items():
        random.shuffle(indexes)
        split_index = int(len(indexes) * 0.9)
        train_indexes.extend(indexes[:split_index])
        test_indexes.extend(indexes[split_index:])

    train = torch.utils.data.Subset(dataset, train_indexes)
    test = torch.utils.data.Subset(dataset, test_indexes)

    train.classes = dataset.classes
    test.classes = dataset.classes

    return train, test

def get_split_by_class(quick: bool, seed=42):
    """Returns a deterministic split of the dataset into train and test sets.

    The classes in train test does not overlap with test set.
    """
    random.seed(seed)
    dataset = get_dataset(quick=quick)

    # Check cache
    path = f"/tmp/labels{'-quick' if quick else ''}.pkl"
    try:
        with open(path, "rb") as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        # Cache miss
        logger.info("Cache miss, loading labels from dataset")
        labels = []
        for idx, (_, label) in enumerate(tqdm(dataset, disable=not sys.stdout.isatty())):
            labels.append(label.item())
        pickle.dump(labels, open(path, "wb"))
        with open(path, "rb") as f:
            labels = pickle.load(f)

    label_to_indexes = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indexes[label].append(idx)
        
    print(max(label_to_indexes, key=lambda k: len(label_to_indexes[k])))
    print(max(label_to_indexes, key=lambda k: len(label_to_indexes[k])))
    print(f"Target train set length: {len(label_to_indexes) * 0.9}")
    print(f"Target test set length: {len(label_to_indexes) * 0.1}")
    
    train_indexes, test_indexes = [], []
    random.shuffle(label_to_indexes)
    # for label, indexes in label_to_indexes.items():
    #     random.shuffle(indexes)
    #     split_index = int(len(indexes) * 0.9)
    #     train_indexes.extend(indexes[:split_index])
    #     test_indexes.extend(indexes[split_index:])

    # train = torch.utils.data.Subset(dataset, train_indexes)
    # test = torch.utils.data.Subset(dataset, test_indexes)

    # train.classes = dataset.classes
    # test.classes = dataset.classes

    # return train, test


if __name__ == "__main__":
    # Print stats on dataset
    dataset = get_dataset()
    print(f"Cardinality: {len(dataset)}")
    print(f"Num Classes: {len(dataset.classes)}")
    
    get_split_by_class()
    
    
