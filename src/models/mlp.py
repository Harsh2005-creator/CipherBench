"""
Multi-Layer Perceptron (MLP) model for cryptographic algorithm identification.
"""

import numpy as np
import keras
from keras import layers, models, ops
from typing import Tuple, Dict, Any


def create_mlp_model(input_dim: int = 10, num_classes: int = 5,
                     hidden_layers: list = [64, 32],
                     dropout_rate: float = 0.3,
                     learning_rate: float = 0.001) -> keras.Model:
    """
    Create Multi-Layer Perceptron model.

    Args:
        input_dim: Number of input features (10 NIST features)
        num_classes: Number of output classes (2 for binary, 5 for multiclass)
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='MLP_CipherBench')

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers with dropout
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

    # Output layer
    if num_classes == 2:
        # Binary classification
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    else:
        # Multiclass classification
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

    return model


def train_mlp(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              verbose: int = 0, validation_split: float = 0.2) -> Tuple[keras.Model, Dict]:
    """
    Train MLP model with early stopping.

    IMPORTANT: Uses validation_split to split X_train into train/validation.
    X_test is NOT used during training - only for final evaluation.

    Args:
        model: Compiled Keras model
        X_train: Training features (will be split into train/val)
        y_train: Training labels (will be split into train/val)
        X_test: Test features (NOT used during training)
        y_test: Test labels (NOT used during training)
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        verbose: Verbosity level
        validation_split: Fraction of training data to use for validation

    Returns:
        Trained model and training history
    """
    # Early stopping callback - monitors validation loss from validation_split
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=verbose
    )

    # Train model - validation_split automatically splits X_train
    # Test set (X_test, y_test) is NOT used here
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,  # Split from X_train only
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose
    )

    return model, history.history


class MLPClassifier:
    """Wrapper class for MLP to provide sklearn-like interface."""

    def __init__(self, num_classes: int = 5, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 verbose: int = 0):
        """
        Initialize MLP classifier.

        Args:
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
        """
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model = None
        self.history = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit the MLP model."""
        input_dim = X_train.shape[1]

        # Create model
        self.model = create_mlp_model(
            input_dim=input_dim,
            num_classes=self.num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )

        # Use validation set if provided, otherwise use training set
        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train

        # Train
        self.model, self.history = train_mlp(
            self.model, X_train, y_train, X_val, y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            # Binary: threshold at 0.5
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multiclass: argmax
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X, verbose=0)

    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            'num_classes': self.num_classes,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
