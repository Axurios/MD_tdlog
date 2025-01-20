import numpy as np
import pytest
from neural_network import MSELoss, MAELoss, CrossEntropyLoss, HingeLoss, SimpleMLP

model = SimpleMLP(in_dimension=3, hidden_dimension=16, num_classes=3)
x = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
labels = np.array([2, 0])  # Correct labels
y, _,_ = model.forward_grid(x,0.1,3)

def test_mse_forward():
    # Test pour la perte MSE (Mean Squared Error)
    mse = MSELoss(num_classes=3)
    loss = mse.forward(x, labels)
    expected_loss = 0.44  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de MSE forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_mae_forward():
    # Test pour la perte MAE (Mean Absolute Error)
    mae = MAELoss(num_classes=3)
    loss = mae.forward(x, labels)
    expected_loss = 1.0  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de MAE forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_hinge_forward():
    # Test pour la perte Hinge
    hinge = HingeLoss(num_classes=3)
    loss = hinge.forward(x, labels)
    expected_loss = 3.0  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de Hinge forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_cel_forward():
    """Test the forward method (cross-entropy loss computation)."""
    cel = CrossEntropyLoss(num_classes=3)

    loss = cel.forward(x, labels)
    
    # Check that loss is computed correctly (manually verify if needed)
    assert loss >= 0, "Cross-entropy loss should be non-negative"

def test_mse_forward_grid():
    mse = MSELoss(num_classes=3)
    result = mse.forward_grid(y, labels)
    assert result.shape == y.shape, "Forward grid output should have the same shape as input grid"

def test_mae_forward_grid():
    mae = MAELoss(num_classes=3)
    result = mae.forward_grid(y, labels)
    assert result.shape == y.shape, "Forward grid output should have the same shape as input grid"


def test_hinge_forward_grid():
    hinge = HingeLoss(num_classes=3)
    result = hinge.forward_grid(y, labels)
    assert result.shape == y.shape, "Forward grid output should have the same shape as input grid"

def test_cel_forward_grid():
    cel = CrossEntropyLoss(num_classes=3)
    result = cel.forward_grid(y, labels)
    assert result.shape == y.shape, "Forward grid output should have the same shape as input grid"

def test_mse_backward():
    mse = MSELoss(num_classes=3)
    grad = mse.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"

def test_mae_backward():
    mae = MAELoss(num_classes=3)
    grad = mae.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"

def test_hinge_backward():
    hinge = HingeLoss(num_classes=3)

    try:
        # Attempt to compute the backward pass directly
        grad = hinge.backward(x, labels)
    except ValueError as e:
        # If forward wasn't called, compute it first
        print(f"Exception caught: {e}. Computing forward...")
        hinge.forward(x, labels)
        grad = hinge.backward(x, labels)

    # Verify the gradient shape matches the input shape
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"
    print("Gradient:\n", grad)

def test_cel_backward():
    """Test the backward method (gradient computation)."""
    cel = CrossEntropyLoss(num_classes=3)
    grads = cel.backward(x, labels)
    
    # Check that gradients have the correct shape
    assert grads.shape == x.shape, "Gradients shape should match logits shape"

def test_cel_make_target():
    """Test the one-hot encoding function."""
    cel = CrossEntropyLoss(num_classes=3)

    target = cel.make_target(x, labels)
    
    # Expected target matrix
    expected_target = np.array([
        [0, 0, 1],  # Class 2 for example 1
        [1, 0, 0]   # Class 0 for example 2
    ])
    
    assert np.array_equal(target, expected_target), "make_target did not return the expected one-hot encoding"

def test_cel_softmax():
    """Test the softmax function."""
    cel = CrossEntropyLoss(num_classes=3)

    probs = cel.softmax(x)
    
    # Check that each row sums to 1
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1), "Softmax outputs do not sum to 1"
    print("test_softmax passed!")
