"""
Unit tests for LogisticRegressor.

Tests cover:
- make_prediction: sigmoid outputs valid probabilities
- loss_function: binary cross-entropy is calculated correctly
- calculate_gradient: gradient has correct shape and direction
- training: weights update and loss decreases over training
"""

import pytest
import numpy as np
from regression.logreg import LogisticRegressor
from regression import utils
from sklearn.preprocessing import StandardScaler


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_model(num_feats=3, seed=0):
    """Return a LogisticRegressor with a fixed random seed for reproducibility."""
    np.random.seed(seed)
    return LogisticRegressor(num_feats=num_feats, learning_rate=0.01, tol=0.001,
                              max_iter=50, batch_size=10)


def padded_X(X):
    """Append bias column (as train_model does internally)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_prediction():
    """
    make_prediction should return values strictly between 0 and 1 (sigmoid range).
    Also checks that extreme positive inputs → ~1 and extreme negative inputs → ~0.
    """
    model = make_model(num_feats=3)

    # Random input matrix with bias column already appended
    np.random.seed(1)
    X = padded_X(np.random.randn(50, 3))

    y_pred = model.make_prediction(X)

    # Output shape matches number of samples
    assert y_pred.shape == (50,), "Prediction shape mismatch."

    # All outputs are valid probabilities
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1), \
        "Sigmoid output must be in [0, 1]."

    # Large positive linear output → prediction close to 1
    model.W = np.ones(4) * 100
    X_pos = padded_X(np.ones((1, 3)))
    assert model.make_prediction(X_pos)[0] > 0.99, \
        "Large positive input should yield prediction near 1."

    # Large negative linear output → prediction close to 0
    model.W = -np.ones(4) * 100
    assert model.make_prediction(X_pos)[0] < 0.01, \
        "Large negative input should yield prediction near 0."


def test_loss_function():
    """
    Binary cross-entropy loss should be:
      - 0 (or near 0) for perfect predictions
      - Higher when predictions are worse
      - ~log(2) ≈ 0.693 when predictions are uniformly 0.5
    """
    model = make_model(num_feats=2)

    # Perfect predictions → loss ≈ 0
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_perfect = np.array([1 - 1e-10, 1e-10, 1 - 1e-10, 1e-10])
    loss_perfect = model.loss_function(y_true, y_perfect)
    assert loss_perfect < 0.01, \
        f"Loss for perfect predictions should be near 0, got {loss_perfect:.4f}."

    # Worst-case predictions (completely wrong) → high loss
    y_worst = np.array([1e-10, 1 - 1e-10, 1e-10, 1 - 1e-10])
    loss_worst = model.loss_function(y_true, y_worst)
    assert loss_worst > loss_perfect, \
        "Worst-case predictions should yield higher loss than perfect predictions."

    # Uniform 0.5 predictions → loss ≈ log(2)
    y_half = np.full(4, 0.5)
    loss_half = model.loss_function(y_true, y_half)
    assert abs(loss_half - np.log(2)) < 0.01, \
        f"Loss for 0.5 predictions should be ~{np.log(2):.4f}, got {loss_half:.4f}."

    # Loss must be non-negative
    assert loss_perfect >= 0 and loss_half >= 0 and loss_worst >= 0, \
        "Loss must always be non-negative."


def test_gradient():
    """
    calculate_gradient should return an array of the same shape as self.W.
    When predictions are too high (y_pred > y_true), gradients should be positive,
    pushing weights down to reduce predictions.
    """
    num_feats = 3
    model = make_model(num_feats=num_feats)

    np.random.seed(2)
    X_raw = np.random.randn(20, num_feats)
    X = padded_X(X_raw)   # shape (20, 4) — includes bias

    # All-zeros labels, force model to predict high values → gradient should be positive
    model.W = np.ones(num_feats + 1) * 10
    y_true = np.zeros(20)

    grad = model.calculate_gradient(y_true, X)

    # Shape matches weights
    assert grad.shape == model.W.shape, \
        f"Gradient shape {grad.shape} does not match weight shape {model.W.shape}."

    # With large positive weights and all-zero labels, most gradient entries should be > 0
    assert np.mean(grad) > 0, \
        "When predictions overshoot (large W, y_true=0), gradient mean should be positive."

    # Gradient should drive loss down — taking a step should reduce the loss
    y_pred_before = model.make_prediction(X)
    loss_before = model.loss_function(y_true, y_pred_before)

    model.W = model.W - 0.1 * grad

    y_pred_after = model.make_prediction(X)
    loss_after = model.loss_function(y_true, y_pred_after)

    assert loss_after < loss_before, \
        "A gradient step should reduce the loss."


def test_training():
    """
    After training, weights should have changed and the final training loss should
    be lower than the initial loss (model should learn something meaningful).
    Uses the real NSCLC dataset with the default feature set.
    """
    # Load and scale data
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    num_feats = X_train.shape[1]
    np.random.seed(0)
    model = LogisticRegressor(
        num_feats=num_feats,
        learning_rate=0.001,
        tol=0.0001,
        max_iter=200,
        batch_size=32
    )

    W_before = model.W.copy()

    model.train_model(X_train, y_train, X_val, y_val)

    W_after = model.W.copy()

    # Weights must have changed
    assert not np.allclose(W_before, W_after), \
        "Weights did not change during training."

    # Loss history must be populated
    assert len(model.loss_hist_train) > 0, \
        "Training loss history is empty after training."
    assert len(model.loss_hist_val) > 0, \
        "Validation loss history is empty after training."

    # Final loss should be lower than the initial loss (model is learning)
    initial_loss = model.loss_hist_train[0]
    final_loss = model.loss_hist_train[-1]
    assert final_loss < initial_loss, \
        f"Final training loss ({final_loss:.4f}) is not less than initial loss ({initial_loss:.4f})."
