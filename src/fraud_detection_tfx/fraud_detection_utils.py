"""
Fraud Detection Pipeline - IEEE-CIS Transactions Dataset

This module includes preprocessing, model training, and utilities for fraud detection using TFX.
"""

from typing import Optional
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer import fn_args_utils
from tfx_bsl.tfxio import dataset_options

# Feature Keys for IEEE-CIS Dataset
_CATEGORICAL_FEATURE_KEYS = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain'
]

_NUMERIC_FEATURE_KEYS = [
    'TransactionAmt', 'C1', 'C2', 'C13', 'D1', 'D15', 'V1', 'V2', 'V3'
]

# Categorical Features: Limits for Encoding
_MAX_CATEGORICAL_FEATURE_VALUES = [5, 5, 5, 100, 100]

# Model Settings
_VOCAB_SIZE = 50  # Vocabulary size for string features
_OOV_SIZE = 3  # Number of out-of-vocabulary buckets for unknown strings
_LABEL_KEY = 'isFraud'

# Helper Functions
def _transformed_name(key: str) -> str:
    """Return the transformed key name."""
    return key + '_xf'


def _fill_in_missing(x: tf.Tensor) -> tf.Tensor:
    """Fill missing values in input tensors."""
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        return tf.sparse.to_dense(x, default_value)
    return x


def _get_raw_feature_spec(schema):
    """Return raw feature specification from schema."""
    return schema_utils.schema_as_feature_spec(schema).feature_spec


# TFX Preprocessing Function
def preprocessing_fn(inputs):
    """Preprocess the features for the IEEE-CIS dataset."""
    outputs = {}

    # Normalize numeric features
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

    # Process categorical features
    for key, max_value in zip(_CATEGORICAL_FEATURE_KEYS, _MAX_CATEGORICAL_FEATURE_VALUES):
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE
        )

    # Process the label
    outputs[_transformed_name(_LABEL_KEY)] = _fill_in_missing(inputs[_LABEL_KEY])

    return outputs


# Input Function
def _input_fn(
    file_pattern: list[str],
    data_accessor: fn_args_utils.DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 256,
) -> tf.data.Dataset:
    """Load and transform the dataset."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    ).repeat()


# Model Builder

def _build_keras_model(hidden_units: Optional[list[int]] = None, learning_rate: float = 0.001) -> tf.keras.Model:
    """Build a Keras model for fraud detection with adjustable learning rate."""
    
    # Create input layers for numerical features
    inputs = {
        key: tf.keras.layers.Input(name=key, shape=(1,), dtype=tf.float32)
        for key in _transformed_name(_NUMERIC_FEATURE_KEYS)
    }
    inputs.update({
        key: tf.keras.layers.Input(name=key, shape=(1,), dtype=tf.int32)
        for key in _transformed_name(_CATEGORICAL_FEATURE_KEYS)
    })

    # Process numerical features
    numeric_inputs = [
        tf.keras.layers.Normalization()(inputs[_transformed_name(key)])
        for key in _NUMERIC_FEATURE_KEYS
    ]
    numeric_layer = tf.keras.layers.concatenate(numeric_inputs)

    # Process categorical features
    categorical_inputs = [
        tf.keras.layers.Embedding(input_dim=max_value, output_dim=8)(inputs[_transformed_name(key)])
        for key, max_value in zip(_CATEGORICAL_FEATURE_KEYS, _MAX_CATEGORICAL_FEATURE_VALUES)
    ]
    categorical_layer = tf.keras.layers.concatenate(categorical_inputs)

    # Combine features
    combined = tf.keras.layers.concatenate([numeric_layer, categorical_layer])

    # Dense Layers
    for units in (hidden_units or [64, 32, 16]):
        combined = tf.keras.layers.Dense(units, activation='relu')(combined)

    # Output Layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Compile the model with a specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Log model summary
    model.summary(print_fn=logging.info)
    
    return model


def train_model(train_dataset, eval_dataset, model, train_steps, eval_steps, model_run_dir, serving_model_dir):
    """
    Train the given Keras model with the provided datasets.

    Args:
    - train_dataset: The training dataset.
    - eval_dataset: The evaluation dataset.
    - model: The Keras model to train.
    - train_steps: The number of steps for training.
    - eval_steps: The number of steps for evaluation.
    - model_run_dir: Directory for model logs.
    - serving_model_dir: Directory to save the trained model.
    """
    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=model_run_dir),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]

    # Train the model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=train_steps,
        validation_steps=eval_steps,
        epochs=10,
        callbacks=callbacks
    )

    # Save the trained model
    tf.saved_model.save(model, serving_model_dir)

# Model Training Function
def run_fn(fn_args: fn_args_utils.FnArgs):
    """Train the fraud detection model, using best hyperparameters from the tuner."""

    # Retrieve the best hyperparameters passed to the trainer
    best_hyperparameters = fn_args.hyperparameters
    
    # Extract the hyperparameters (e.g., hidden_units, learning_rate)
    hidden_units = best_hyperparameters.get('hidden_units', [64, 32, 16])  # Default to [64, 32, 16] if not provided
    learning_rate = best_hyperparameters.get('learning_rate', 0.001)  # Default to 0.001 if not provided

    # Prepare the transformed data for training and evaluation
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output, batch_size=256
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output, batch_size=256
    )

    # Build the model with hyperparameters passed from the tuner
    model = _build_keras_model(hidden_units=hidden_units, learning_rate=learning_rate)

    # Train the model
    train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        train_steps=fn_args.train_steps,
        eval_steps=fn_args.eval_steps,
        model_run_dir=fn_args.model_run_dir,
        serving_model_dir=fn_args.serving_model_dir
    )



def stats_options_updater_fn(unused_stats_type, stats_options):
    """Configure stats options."""
    return stats_options
