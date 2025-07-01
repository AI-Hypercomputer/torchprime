import model


def test_model_deterministism():
    # Arrange
    num_classes = 1090
    num_channels = 3
    seed = 42

    # Act
    model1, hash1 = model.get_model(num_classes, num_channels, seed)
    model2, hash2 = model.get_model(num_classes, num_channels, seed)

    # Assert
    assert hash1 == hash2, "Models are not equal"


def test_model_deterministism_different():
    # Arrange
    num_classes = 1090
    num_channels = 3
    seed1 = 0
    seed2 = 42

    # Act
    model1, hash1 = model.get_model(num_classes, num_channels, seed1)
    model2, hash2 = model.get_model(num_classes, num_channels, seed2)

    # Assert
    assert hash1 != hash2, "Models are equal"


def test_freeze_unfreeze():
    def classify_params(model):
        frozen = set()
        unfrozen = set()
        for name, param in model1.named_parameters():
            if param.requires_grad:
                unfrozen.add(name)
            else:
                frozen.add(name)
        return frozen, unfrozen

    """Test that the model can be frozen and unfrozen correctly."""
    # Arrange
    num_classes = 1090
    num_channels = 3
    seed = 42

    # Act
    model1, _ = model.get_model(num_classes, num_channels, seed)
    model.freeze_backbone(model1)

    # Assert
    frozen, unfrozen = classify_params(model1)
    assert unfrozen == {
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "fc.weight",
        "fc.bias",
    }, "Incorrect frozen parameters"
    assert len(frozen) > 0

    # Act
    model.unfreeze_all(model1)

    # Assert
    frozen, unfrozen = classify_params(model1)
    assert len(frozen) == 0, "Some parameters are still frozen"
    assert len(unfrozen) > 0
