"""
Modul ini mengatur fungsi untuk tuning hyperparameter
model klasifikasi biner menggunakan Keras Tuner.
"""

import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
from transform import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, transformed_name
from trainer import input_fn

# Fungsi untuk membuat input features (menghindari duplikasi kode)
def create_input_features():
    """
    Membuat input features untuk numerical dan categorical features.
    
    Returns:
        list: Daftar input layers.
    """
    input_features = []

    # Input layers for numerical features
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Input layers for categorical features
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    return input_features

# Fungsi untuk membuat model
def model_builder(hyperparameters):
    """
    Defines and returns a Keras model for binary classification.
    """
    # Membuat input features
    input_features = create_input_features()

    # Concatenate all inputs
    concatenate = tf.keras.layers.concatenate(input_features)

    # Hidden layers with hyperparameter tuning
    deep = tf.keras.layers.Dense(
        hyperparameters.Choice('units_layer1', [64, 128, 256]),
        activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(
        hyperparameters.Choice('dropout_layer1', [0.2, 0.3, 0.4]))(deep)

    deep = tf.keras.layers.Dense(
        hyperparameters.Choice('units_layer2', [32, 64, 128]),
        activation="relu")(deep)
    deep = tf.keras.layers.Dropout(
        hyperparameters.Choice('dropout_layer2', [0.2, 0.3, 0.4]))(deep)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    # Compile the model
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparameters.Choice('learning_rate', [0.001, 0.0001])
        ),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

# Fungsi tuner
def tuner_fn(fn_args: FnArgs):
    """
    Hyperparameter tuning function for the model.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Load training and evaluation datasets
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    # Create tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=10,
        directory=fn_args.working_dir,
        project_name='attrition_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 10
        }
    )
