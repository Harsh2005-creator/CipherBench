"""
1D Convolutional Neural Network (1D-CNN) model for cryptographic
algorithm identification.

Binary labels are normalized internally to 0/1 and mapped back to the
original labels during prediction. The final test set is never used for
training, validation, early stopping, or model selection.
"""

import numpy as np
import keras
from keras import layers, models
from typing import Tuple, Dict, Any


def create_cnn_model(
    input_dim: int = 10,
    num_classes: int = 5,
    filters: list = None,
    kernel_size: int = 3,
    pool_size: int = 2,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    if filters is None:
        filters = [32, 64]

    model = models.Sequential(name="CNN_1D_CipherBench")
    model.add(layers.Input(shape=(input_dim, 1)))

    for i, f in enumerate(filters):
        model.add(
            layers.Conv1D(
                filters=f,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                name=f"conv1d_{i+1}",
            )
        )
        model.add(layers.BatchNormalization(name=f"batchnorm_{i+1}"))

        if input_dim // (pool_size ** (i + 1)) >= 1:
            model.add(layers.MaxPooling1D(pool_size=pool_size, name=f"maxpool_{i+1}"))

        model.add(layers.Dropout(dropout_rate, name=f"dropout_conv_{i+1}"))

    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(64, activation="relu", name="dense_hidden"))
    model.add(layers.Dropout(dropout_rate, name="dropout_dense"))

    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
    else:
        model.add(layers.Dense(num_classes, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def train_cnn(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
    validation_split: float = 0.2,
) -> Tuple[keras.Model, Dict]:
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
    if X_val is not None and X_val.ndim == 2:
        X_val = np.expand_dims(X_val, axis=-1)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=verbose,
    )

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            **fit_kwargs,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            **fit_kwargs,
        )

    return model, history.history


class CNN1DClassifier:
    """Sklearn-like 1D-CNN wrapper with safe label handling."""

    def __init__(
        self,
        num_classes: int = 5,
        filters: list = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.num_classes = num_classes
        self.filters = [32, 64] if filters is None else filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.history = None
        self.classes_ = None

    def _encode_labels(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        if fit:
            self.classes_ = np.unique(y)
            if len(self.classes_) != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} classes, found {len(self.classes_)}: "
                    f"{self.classes_}"
                )

        if self.classes_ is None:
            raise ValueError("Model has not been fitted.")

        mapping = {label: i for i, label in enumerate(self.classes_)}
        try:
            return np.asarray([mapping[v] for v in y], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Unknown label {exc.args[0]} encountered after fitting.")

    def _decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted.")
        return self.classes_[np.asarray(encoded, dtype=int)]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        keras.utils.set_random_seed(self.random_state)

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train_encoded = self._encode_labels(y_train, fit=True)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val_encoded = self._encode_labels(y_val, fit=False)
        else:
            y_val_encoded = None

        self.model = create_cnn_model(
            input_dim=X_train.shape[1],
            num_classes=self.num_classes,
            filters=self.filters,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
        )

        self.model, self.history = train_cnn(
            self.model,
            X_train,
            y_train_encoded,
            X_val,
            y_val_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)

        predictions = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            encoded = (predictions.reshape(-1) >= 0.5).astype(int)
        else:
            encoded = np.argmax(predictions, axis=1)

        return self._decode_labels(encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)

        probabilities = self.model.predict(X, verbose=0)
        if self.num_classes == 2:
            p = probabilities.reshape(-1, 1)
            return np.hstack([1.0 - p, p])
        return probabilities

    def get_params(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "dropout_rate": self.dropout_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
