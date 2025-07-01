import torch

import data


def test_get_dataset():
    # Arrange
    expected_num_classes = 1090
    expected_cardinality = 209243

    # Act
    dataset = data.get_dataset()

    # Assert
    assert len(dataset.classes) == expected_num_classes, (
        f"Expected {expected_num_classes} classes, but got {len(dataset.classes)}"
    )
    assert len(dataset) == expected_cardinality, (
        f"Expected {expected_cardinality} samples, but got {len(dataset)}"
    )


def test_elements():
    # Arrange
    dataset = data.get_dataset()

    # Act
    image, label = dataset[0]
    print(dataset[0])

    # Assert
    assert image.size() == (3, 224, 224), (
        f"Expected image size (3, 224, 224), but got {image.size()}"
    )
    assert isinstance(label, torch.Tensor), (
        f"Expected label to be a tensor, but got {type(label)}"
    )
    assert 0 <= label < len(dataset.classes), (
        f"Label {label} is out of bounds for {len(dataset.classes)} classes"
    )
