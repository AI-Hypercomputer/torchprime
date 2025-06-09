import pytest
import torch
import torch.nn as nn

from torchprime.torch_xla_models.model_rewriting.auto_trace import auto_trace


class TestAutoTrace:
  @pytest.fixture
  def mock_trace(self, monkeypatch):
    """Mock xp.Trace with a context manager that records calls"""
    trace_calls = []

    # Create a class that acts as both callable and context manager
    class MockTrace:
      calls: list[str]

      def __init__(self, name):
        self.name = name
        trace_calls.append(name)

      def __enter__(self):
        return self

      def __exit__(self, *args):
        pass

    MockTrace.calls = trace_calls
    monkeypatch.setattr("torch_xla.debug.profiler.Trace", MockTrace)
    return MockTrace

  def test_linear_layers_are_traced(self, mock_trace):
    """Test that Linear layers get traced with their attribute names"""

    class Model(nn.Module):
      def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.relu = nn.ReLU()  # Should not be traced

      def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # Clear any setup calls
    mock_trace.calls.clear()

    # Create model and run forward
    model = auto_trace(Model())
    x = torch.randn(5, 10)
    output = model(x)

    # Verify traces were created with correct names
    assert "fc1" in mock_trace.calls
    assert "fc2" in mock_trace.calls
    assert len(mock_trace.calls) == 2

    # Verify output shape is correct
    assert output.shape == (5, 30)

  def test_custom_traced_types(self, mock_trace):
    """Test tracing custom module types"""

    class CustomLayer(nn.Module):
      def forward(self, x):
        return x * 2

    class Model(nn.Module):
      def __init__(self):
        super().__init__()
        self.custom = CustomLayer()
        self.linear = nn.Linear(10, 10)  # Should not be traced

      def forward(self, x):
        x = self.custom(x)
        x = self.linear(x)
        return x

    mock_trace.calls.clear()

    model = auto_trace(Model(), traced_types=(CustomLayer,))
    x = torch.randn(5, 10)
    _ = model(x)

    # Only custom layer should be traced
    assert "custom" in mock_trace.calls
    assert "linear" not in mock_trace.calls

  def test_multiple_forward_calls(self, mock_trace):
    """Test that traces are created on each forward call"""

    class Model(nn.Module):
      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

      def forward(self, x):
        return self.fc(x)

    mock_trace.calls.clear()

    model = auto_trace(Model())
    x = torch.randn(5, 10)

    # Call forward multiple times
    for _ in range(3):
      model(x)

    # Should have 3 traces
    assert mock_trace.calls.count("fc") == 3

  def test_nested_modules_recursive(self, mock_trace):
    """Test tracing in nested module structures"""

    class SubModule(nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

      def forward(self, x):
        return self.linear(x)

    class Model(nn.Module):
      def __init__(self):
        super().__init__()
        self.sub = SubModule()
        self.fc = nn.Linear(10, 10)

      def forward(self, x):
        x = self.sub(x)
        x = self.fc(x)
        return x

    mock_trace.calls.clear()

    model = Model()
    model = auto_trace(model)
    x = torch.randn(5, 10)
    _ = model(x)

    assert "fc" in mock_trace.calls
    assert "sub" not in mock_trace.calls  # Only `Linear` layer should be traced
    assert "linear" in mock_trace.calls
    assert len(mock_trace.calls) == 2

  def test_original_functionality_preserved(self, mock_trace):
    """Test that the original forward functionality is preserved"""

    # Create two identical models, one with decorator
    class MyModel(nn.Module):
      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

      def forward(self, x):
        return self.fc(x) * 2

    # Set same weights
    traced = auto_trace(MyModel())
    untraced = MyModel()
    untraced.fc.weight.data = traced.fc.weight.data.clone()
    untraced.fc.bias.data = traced.fc.bias.data.clone()

    x = torch.randn(5, 10)

    # Outputs should be identical
    traced_out = traced(x)
    untraced_out = untraced(x)

    torch.testing.assert_close(traced_out, untraced_out)
