"""
Multi-Layer Perceptron (MLP) model for cryptographic algorithm identification.

Binary labels are normalized internally to 0/1 and mapped back to the
original labels during prediction. The final test set is never used for
training, validation, early stopping, or model selection.
"""

import numpy as np
import keras
from keras import layers, models
from typing import Tuple, Dict, Any


def create_mlp_model(
    input_dim: int = 10,
    num_classes: int = 5,
    hidden_layers: list = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    if hidden_layers is None:
        hidden_layers = [64, 32]

    model = models.Sequential(name="MLP_CipherBench")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

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


def train_mlp(
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
    """
    Train using either an explicit validation set or a split from X_train.

    X_test is deliberately not accepted here: the test set must only be used
    by the caller for final evaluation.
    """
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


class MLPClassifier:
    """Sklearn-like MLP wrapper with safe label handling."""

    def __init__(
        self,
        num_classes: int = 5,
        hidden_layers: list = None,
        dropout_rate: float = 0.3,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: int = 0,
    ):
        self.num_classes = num_classes
        self.hidden_layers = [64, 32] if hidden_layers is None else hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
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
        encoded = np.asarray(encoded, dtype=int)
        return self.classes_[encoded]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train_encoded = self._encode_labels(y_train, fit=True)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val_encoded = self._encode_labels(y_val, fit=False)
        else:
            y_val_encoded = None

        self.model = create_mlp_model(
            input_dim=X_train.shape[1],
            num_classes=self.num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
        )

        self.model, self.history = train_mlp(
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

        predictions = self.model.predict(
            np.asarray(X, dtype=np.float32), verbose=0
        )

        if self.num_classes == 2:
            encoded = (predictions.reshape(-1) >= 0.5).astype(int)
        else:
            encoded = np.argmax(predictions, axis=1)

        return self._decode_labels(encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        probabilities = self.model.predict(
            np.asarray(X, dtype=np.float32), verbose=0
        )
        if self.num_classes == 2:
            p = probabilities.reshape(-1, 1)
            return np.hstack([1.0 - p, p])
        return probabilities

    def get_params(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
