"""Tests for model architecture instantiation and forward pass."""

import pytest


def test_resnet50_output_shape():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import resnet50
    import torch.nn as nn

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.eval()

    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_efficientnet_b0_output_shape():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import efficientnet_b0
    import torch.nn as nn

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.eval()

    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_densenet121_output_shape():
    torch = pytest.importorskip("torch")
    from torchvision.models import densenet121
    import torch.nn as nn

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model.eval()

    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)
