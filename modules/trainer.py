"""
Modul ini berisi fungsi untuk pelatihan model machine learning
menggunakan pipeline TFX, termasuk definisi model, proses pelatihan,
dan evaluasi.
"""

import os
import tensorflow as tf
import tensorflow_transform as tft
from keras.utils.vis_utils import plot_model


from transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    transformed_name,
    LABEL_KEY,
)

# Fungsi untuk membuat model
def get_model(show_summary=True):
    """
    Defines a Keras model for binary classification.
    """
    # Input layers for numerical features
    input_features = []
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Input layers for categorical features
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Concatenate all inputs
    concatenate = tf.keras.layers.concatenate(input_features)

    # Hidden layers
    deep = tf.keras.layers.Dense(128, activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(0.3)(deep)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    # Compile the model
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    if show_summary:
        model.summary()

    return model

# Fungsi untuk membaca data yang sudah dikompresi
def gzip_reader_fn(filenames):
    """Reads compressed data."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Fungsi untuk mendapatkan fitur yang sudah ditransformasi
def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a serving function for the model."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Parses a serialized tf.Example and returns model predictions."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

# Fungsi untuk membuat dataset
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for training/evaluation."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

# Fungsi untuk menjalankan pelatihan model
def run_fn(fn_args):
    """
    Train the model using the given arguments.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Load training and evaluation datasets
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    # Define the model
    model = get_model()

    # Define callbacks
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    # Save the model with serving signature
    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

    # Visualize the model architecture
    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
