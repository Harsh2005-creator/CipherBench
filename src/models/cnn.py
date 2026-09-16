"""
1D Convolutional Neural Network (1D-CNN) model for cryptographic algorithm identification.
Reshapes the 10-feature NIST data to 1D inputs to exploit spatial/ordered relationships in features.
"""

import numpy as np
import keras
from keras import layers, models, ops
from typing import Tuple, Dict, Any


def create_cnn_model(input_dim: int = 10, num_classes: int = 5,
                     filters: list = [32, 64],
                     kernel_size: int = 3,
                     pool_size: int = 2,
                     dropout_rate: float = 0.3,
                     learning_rate: float = 0.001) -> keras.Model:
    """
    Create a 1D CNN model.

    Args:
        input_dim: Number of input features (10 NIST features)
        num_classes: Number of output classes (2 for binary, 5 for multiclass)
        filters: List of filter sizes for Conv1D layers
        kernel_size: Convolution kernel size
        pool_size: MaxPooling size
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='CNN_1D_CipherBench')

    # Input layer representing reshaped input (10 features, 1 channel)
    model.add(layers.Input(shape=(input_dim, 1)))

    # Conv1D layers block
    for i, f in enumerate(filters):
        model.add(layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'batchnorm_{i+1}'))

        # Only pool if sequence length allows it, otherwise skip to keep dimension positive
        # Sequence length starts at 10. Maxpooling splits it by pool_size.
        if input_dim // (pool_size ** (i + 1)) > 0:
            model.add(layers.MaxPooling1D(pool_size=pool_size, name=f'maxpool_{i+1}'))

        model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))

    # Flatten and Dense classification layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(64, activation='relu', name='dense_hidden'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))

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


def train_cnn(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              verbose: int = 0) -> Tuple[keras.Model, Dict]:
    """
    Train 1D CNN model.

    Args:
        model: Compiled Keras model
        X_train: Training features, already reshaped or will be reshaped here
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epochs: Max epochs
        batch_size: Batch size
        verbose: Verbosity

    Returns:
        Trained model & training history
    """
    # Reshape features to (Samples, Time Steps, Channels) if not already
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_test.shape) == 2:
        X_test = np.expand_dims(X_test, axis=-1)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=verbose
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose
    )

    return model, history.history


class CNN1DClassifier:
    """Wrapper class for 1D-CNN providing sklearn-like interface."""

    def __init__(self, num_classes: int = 5, filters: list = [32, 64],
                 kernel_size: int = 3, pool_size: int = 2,
                 dropout_rate: float = 0.3, epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 verbose: int = 0):
        """
        Initialize CNN classifier.
        """
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model = None
        self.history = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit the CNN model."""
        input_dim = X_train.shape[1]

        # Reshape data to (Samples, Features, 1)
        X_train_reshaped = np.expand_dims(X_train, axis=-1)

        if X_val is None or y_val is None:
            X_val_reshaped, y_val = X_train_reshaped, y_train
        else:
            X_val_reshaped = np.expand_dims(X_val, axis=-1)

        # Create model
        self.model = create_cnn_model(
            input_dim=input_dim,
            num_classes=self.num_classes,
            filters=self.filters,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )

        # Train
        self.model, self.history = train_cnn(
            self.model, X_train_reshaped, y_train,
            X_val_reshaped, y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_reshaped = np.expand_dims(X, axis=-1)
        predictions = self.model.predict(X_reshaped, verbose=0)

        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_reshaped = np.expand_dims(X, axis=-1)
        return self.model.predict(X_reshaped, verbose=0)

    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            'num_classes': self.num_classes,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
