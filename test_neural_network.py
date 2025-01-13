import numpy as np
import pytest
from neural_network import MSE, MAELoss, HuberLoss, HingeLoss 


x = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
labels = np.array([2, 0])  # Correct labels
x_grid = np.array([[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]])

def test_mse_forward():
    # Test pour la perte MSE (Mean Squared Error)
    mse = MSE(num_classes=3)
    loss = mse.forward(x, labels)
    expected_loss = 0.44  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de MSE forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_mae_forward():
    # Test pour la perte MAE (Mean Absolute Error)
    mae = MAELoss(num_classes=3)
    loss = mae.forward(x, labels)
    expected_loss = 0.36666666666666664  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de MAE forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_huber_forward():
    # Test pour la perte de Huber
    huber = HuberLoss(delta=1.0, num_classes=3)
    loss = huber.forward(x, labels)
    expected_loss = 0.245  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de Huber forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_hinge_forward():
    # Test pour la perte Hinge
    hinge = HingeLoss(num_classes=3)
    loss = hinge.forward(x, labels)
    expected_loss = 0.95  # Résultat attendu calculé manuellement
    assert isinstance(loss, float), "La sortie de Hinge forward doit être un flottant"
    assert loss >= 0, "La perte doit être non négative"
    assert np.isclose(loss, expected_loss, atol=1e-6), f"Perte attendue {expected_loss}, mais obtenu {loss}"

def test_mse_forward_grid():
    mse = MSE(num_classes=3)
    result = mse.forward_grid(x_grid, labels)
    assert result.shape == x_grid.shape, "Forward grid output should have the same shape as input grid"

def test_mae_forward_grid():
    mae = MAELoss(num_classes=3)
    result = mae.forward_grid(x_grid, labels)
    assert result.shape == x_grid.shape, "Forward grid output should have the same shape as input grid"

def test_huber_forward_grid():
    huber = HuberLoss(delta=1.0, num_classes=3)
    result = huber.forward_grid(x_grid, labels)
    assert result.shape == x_grid.shape, "Forward grid output should have the same shape as input grid"

def test_hinge_forward_grid():
    hinge = HingeLoss(num_classes=3)
    result = hinge.forward_grid(x_grid, labels)
    assert result.shape == x_grid.shape, "Forward grid output should have the same shape as input grid"

def test_mse_backward():
    mse = MSE(num_classes=3)
    grad = mse.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"

def test_mae_backward():
    mae = MAELoss(num_classes=3)
    grad = mae.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"

def test_huber_backward():
    huber = HuberLoss(delta=1.0, num_classes=3)
    grad = huber.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"

def test_hinge_backward():
    hinge = HingeLoss(num_classes=3)
    grad = hinge.backward(x, labels)
    assert grad.shape == x.shape, "Backward gradient shape should match input shape"
